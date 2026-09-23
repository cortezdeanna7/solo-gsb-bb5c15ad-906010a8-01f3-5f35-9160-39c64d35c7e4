"""CIDR decision map for a *saved* analysis version (read-only).

Given a saved analysis result and one strict IPv4/IPv6 query CIDR, the query
range is carved into the minimum set of pairwise disjoint, jointly exhaustive
CIDR blocks on which the final decision is constant.  Every block reports:

* ``address_count``  : addresses in the block (exact, also for IPv6);
* ``action``         : the effective action ``permit`` / ``deny``, or ``none``
                       when no rule of this version matches the block;
* ``winner_rule_id`` / ``winner_source_tag`` : the rule that wins LPM for the
                       block (``null`` for unmatched blocks);
* ``match_chain``    : every rule of the version whose CIDR contains the block,
                       ordered by prefix length (longest first); rules sharing
                       one CIDR are ordered by priority (smaller first) and
                       then by original submission order.  The winning rule is
                       the first entry whenever the block matches at least one
                       rule; the losing same-CIDR duplicates are included.
                       Unmatched blocks carry an empty chain.

Geometry
--------
The version's winning-rule CIDRs of the query's address family form the same
containment trie the analyzer built (nearest enclosing CIDR as parent):

* nodes inside the query are *local*; the local nodes without a local parent
  are the roots that partition the query;
* nodes strictly containing the query are *enclosing*; the closest one is the
  fallback winner (and the chain head) for query gaps that hold no local rule.

The query is partitioned recursively: each node owns ``cidr(node)`` minus its
local children's CIDRs (its effective range, exactly as in the analysis); the
child CIDRs are recursed into.  A root gap is won by the closest enclosing
node; only when no enclosing CIDR exists does the gap match no rule (``none``).

Adjacent sibling blocks with the *same winning rule* (equivalently, both
unmatched) are then collapsed with ``ipaddress.collapse_addresses``.  Blocks
whose actions happen to agree but whose winning rules differ are never merged.
Everything is geometric — addresses are never enumerated, so IPv6 /0 queries
(2**128 addresses) are handled exactly.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any, Iterable

# Reuse the engine's geometry helpers rather than re-implementing subtraction.
from .analyzer import _sort_networks, _subtract

Network = ipaddress._BaseNetwork


class DecisionMapError(ValueError):
    """The query CIDR is not a strict IPv4/IPv6 CIDR (-> HTTP 400)."""


def parse_query_cidr(value: Any) -> Network:
    """Parse a strict query CIDR: a syntactically valid IPv4/IPv6 network with
    no host bits set (``10.0.0.5/24`` is rejected, as for submitted rules).
    """
    if not isinstance(value, str) or not value.strip():
        raise DecisionMapError("cidr query parameter is required")
    text = value.strip()
    try:
        return ipaddress.ip_network(text, strict=True)
    except ValueError as exc:
        raise DecisionMapError(f"invalid CIDR {value!r} ({exc})") from exc


@dataclass(eq=False)
class _Node:
    """One distinct winning-rule CIDR of the version (same-CIDR duplicates
    grouped), restricted to the query's address family."""

    network: Network
    #: Same-CIDR rules in arbitration order: (priority, submission index, record).
    members: list[tuple[int, int, dict[str, Any]]]
    children: list["_Node"] = field(default_factory=list)
    parent: "_Node | None" = None

    @property
    def winner(self) -> dict[str, Any]:
        return self.members[0][2]


@dataclass(eq=False)
class _Cell:
    network: Network
    #: Winning node whose effective range the cell belongs to; for a root gap
    #: this is the closest enclosing node, or ``None`` when no rule matches.
    owner: _Node | None


def _build_family_trie(
    result: dict[str, Any], query: Network
) -> tuple[list[_Node], _Node | None]:
    """Build the containment trie of every rule CIDR in the query's family.

    Returns ``(local_nodes, closest_enclosing)`` where *local* nodes lie inside
    the query and *closest_enclosing* is the longest-prefix node strictly
    containing the query (``None`` when there is none).
    """
    groups: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
    for rule in result.get("rules", []):
        network = ipaddress.ip_network(rule["cidr"])
        if network.version != query.version:
            continue
        groups.setdefault(str(network), []).append(
            (int(rule["priority"]), int(rule["index"]), rule)
        )

    nodes = [
        _Node(network=ipaddress.ip_network(cidr), members=sorted(members))
        for cidr, members in groups.items()
    ]
    nodes.sort(key=lambda n: (int(n.network.network_address), n.network.prefixlen))

    # Nearest enclosing node -> parent (over all family nodes; nodes disjoint
    # from the query can neither enclose nor be enclosed by a relevant one).
    for node in nodes:
        net = node.network
        best: _Node | None = None
        best_prefix = -1
        for candidate in nodes:
            if candidate is node:
                continue
            cnet = candidate.network
            if cnet.prefixlen < net.prefixlen and net.subnet_of(cnet):
                if cnet.prefixlen > best_prefix:
                    best = candidate
                    best_prefix = cnet.prefixlen
        node.parent = best
        if best is not None:
            best.children.append(node)

    for node in nodes:
        node.children.sort(
            key=lambda n: (int(n.network.network_address), n.network.prefixlen)
        )

    local = [n for n in nodes if n.network.subnet_of(query)]
    enclosing = [
        n for n in nodes if query.subnet_of(n.network) and n.network != query
    ]
    closest_enclosing = max(enclosing, key=lambda n: n.network.prefixlen, default=None)
    return local, closest_enclosing


def _carve_cells(
    query: Network, local: list[_Node], fallback: _Node | None
) -> list[_Cell]:
    """Recursively partition the query into decision-constant cells.

    A node owns its CIDR minus its local children's CIDRs (its effective
    range); each child CIDR is recursed into.  A gap outside every local root
    but inside the closest enclosing CIDR is won by that enclosing node (it is
    part of the enclosing rule's effective surface); with no enclosing CIDR
    the gap matches no rule.
    """
    cells: list[_Cell] = []
    local_set = set(local)
    roots = [n for n in local if n.parent not in local_set]

    def visit(region: Network, owner: _Node, children: list[_Node]) -> None:
        child_nets = _sort_networks(c.network for c in children)
        for remainder in _sort_networks(_subtract(region, child_nets)):
            cells.append(_Cell(remainder, owner))
        for child in children:
            visit(child.network, child, child.children)

    root_nets = _sort_networks(n.network for n in roots)
    for gap in _sort_networks(_subtract(query, root_nets)):
        cells.append(_Cell(gap, fallback))
    for root in roots:
        visit(root.network, root, root.children)
    return cells


def _chain_entries(cell: _Cell) -> list[dict[str, Any]]:
    """All rules whose CIDR contains the cell, longest prefix first.

    The walk starts at the owning node — the LPM winner, which for a root gap
    is the closest enclosing node — and follows parent links, so enclosing
    supernets such as a ``10.0.0.0/8`` ancestor are included for every cell.
    Same-CIDR rules follow priority then submission order (the winner first);
    every entry is tagged with its arbitration role on that CIDR.
    """
    chain: list[tuple[int, int, int, dict[str, Any], str]] = []
    node = cell.owner
    while node is not None:
        for pos, (_prio, _idx, rule) in enumerate(node.members):
            chain.append(
                (
                    -node.network.prefixlen,
                    pos,
                    int(rule["index"]),
                    rule,
                    "winner" if pos == 0 else "duplicate",
                )
            )
        node = node.parent
    chain.sort(key=lambda item: (item[0], item[1], item[2]))

    return [
        {
            "rule_id": rule["rule_id"],
            "cidr": rule["cidr"],
            "action": rule["action"],
            "priority": rule["priority"],
            "source_tag": rule["source_tag"],
            "role": role,
        }
        for _neg_len, _pos, _index, rule, role in chain
    ]


def _collapse_same_winner(
    cells: Iterable[_Cell],
) -> list[tuple[Network, dict[str, Any] | None, _Cell]]:
    """Collapse adjacent equal-prefix sibling cells won by the same rule
    (including adjacent unmatched cells).  Returns ``(network, winner,
    representative_cell)`` triples in canonical address order.

    Cells sharing a winning node also share one identical match chain (the
    chain depends only on trie ancestry); unmatched cells likewise share the
    enclosing fallback chain or have none, so any surviving representative
    describes the whole collapsed block.
    """
    groups: dict[object, list[_Cell]] = {}

    def _winner_key(cell: _Cell) -> object:
        return ("w", cell.owner.winner["rule_id"]) if cell.owner is not None else ("n",)

    for cell in cells:
        groups.setdefault(_winner_key(cell), []).append(cell)

    collapsed: list[tuple[Network, dict[str, Any] | None, _Cell]] = []
    for group_cells in groups.values():
        representative = group_cells[0]
        winner = representative.owner.winner if representative.owner is not None else None
        for network in ipaddress.collapse_addresses(
            _sort_networks(c.network for c in group_cells)
        ):
            collapsed.append((network, winner, representative))
    collapsed.sort(
        key=lambda item: (item[0].version, int(item[0].network_address), item[0].prefixlen)
    )
    return collapsed


def build_decision_map(result: dict[str, Any], query: Network) -> dict[str, Any]:
    """Compute the read-only decision map of a saved version for ``query``."""
    local, closest_enclosing = _build_family_trie(result, query)
    cells = _carve_cells(query, local, closest_enclosing)

    blocks: list[dict[str, Any]] = []
    permit_addrs = deny_addrs = none_addrs = 0
    for network, winner, representative in _collapse_same_winner(cells):
        count = network.num_addresses
        if winner is None:
            action: str | None = None
            rule_id: str | None = None
            source_tag: str | None = None
            none_addrs += count
        else:
            action = str(winner["action"])
            rule_id = winner["rule_id"]
            source_tag = winner["source_tag"]
            if action == "permit":
                permit_addrs += count
            else:
                deny_addrs += count
        blocks.append(
            {
                "cidr": str(network),
                "address_count": count,
                "action": action,
                "winner_rule_id": rule_id,
                "winner_source_tag": source_tag,
                "match_chain": _chain_entries(representative),
            }
        )

    return {
        "analysis_id": result.get("id"),
        "ruleset_hash": result.get("ruleset_hash"),
        "analyzer_version": result.get("analyzer_version"),
        "query_cidr": str(query),
        "ip_version": query.version,
        "total_addresses": query.num_addresses,
        "blocks": blocks,
        "summary": {
            "block_count": len(blocks),
            "matched_blocks": sum(1 for b in blocks if b["action"] is not None),
            "unmatched_blocks": sum(1 for b in blocks if b["action"] is None),
            "matched_addresses": permit_addrs + deny_addrs,
            "permit_addresses": permit_addrs,
            "deny_addresses": deny_addrs,
            "unmatched_addresses": none_addrs,
        },
    }
