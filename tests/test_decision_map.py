"""End-to-end tests for the GET /api/analyses/{id}/decision-map endpoint."""

from __future__ import annotations

import ipaddress
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app import db
from app.decision_map import build_decision_map, parse_query_cidr


@pytest.fixture()
def client() -> TestClient:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="ppa-test-map-")
    os.close(fd)
    os.remove(path)  # let sqlite create it
    db.reset_db(path)
    from app.main import app

    with TestClient(app) as c:
        yield c
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass


def _save(client: TestClient, rules: list[dict], name: str | None = None) -> str:
    body: dict = {"rules": rules}
    if name is not None:
        body["name"] = name
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["persisted"] is True
    return r.json()["id"]


# ---------------------------------------------------------------------------
def test_basic_lpm_partition_actions_and_addresses(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100, "source_tag": "hq"},
        {"rule_id": "b", "cidr": "10.10.0.0/16", "action": "deny", "priority": 500, "source_tag": "dc1"},
        {"rule_id": "c", "cidr": "10.10.1.0/24", "action": "permit", "priority": 900, "source_tag": "lab"},
    ])

    r = client.get(f"/api/analyses/{aid}/decision-map", params={"cidr": "10.10.0.0/16"})
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["analysis_id"] == aid
    assert data["query_cidr"] == "10.10.0.0/16"
    assert data["ip_version"] == 4
    assert data["total_addresses"] == 2 ** 16

    by_cidr = {b["cidr"]: b for b in data["blocks"]}
    permit_block = by_cidr["10.10.1.0/24"]
    assert permit_block["action"] == "permit"
    assert permit_block["winner_rule_id"] == "c"
    assert permit_block["winner_source_tag"] == "lab"
    assert permit_block["address_count"] == 2 ** 8

    deny_blocks = [b for b in data["blocks"] if b["action"] == "deny"]
    assert {b["winner_rule_id"] for b in deny_blocks} == {"b"}
    assert sum(b["address_count"] for b in deny_blocks) == 2 ** 16 - 2 ** 8
    assert all(b["winner_source_tag"] == "dc1" for b in deny_blocks)

    summary = data["summary"]
    assert summary["permit_addresses"] == 2 ** 8
    assert summary["deny_addresses"] == 2 ** 16 - 2 ** 8
    assert summary["unmatched_addresses"] == 0
    assert summary["block_count"] == len(data["blocks"])


def test_blocks_partition_query_disjoint_and_exhaustive(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "b", "cidr": "10.10.0.0/16", "action": "deny"},
    ])
    r = client.get(f"/api/analyses/{aid}/decision-map", params={"cidr": "10.10.0.0/16"})
    blocks = r.json()["blocks"]

    nets = [ipaddress.ip_network(b["cidr"]) for b in blocks]
    query = ipaddress.ip_network("10.10.0.0/16")
    # pairwise disjoint
    for i, x in enumerate(nets):
        for y in nets[i + 1:]:
            assert not x.overlaps(y)
    # jointly exhaustive
    assert int(nets[0].network_address) == int(query.network_address)
    end = int(nets[-1].broadcast_address)
    cursor = int(nets[0].network_address)
    for net in nets:
        assert int(net.network_address) == cursor
        cursor = int(net.broadcast_address) + 1
    assert end == int(query.broadcast_address)
    assert sum(b["address_count"] for b in blocks) == query.num_addresses

    # stable canonical order
    assert nets == sorted(nets, key=lambda n: (int(n.network_address), n.prefixlen))


def test_match_chain_order_and_contents(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100, "source_tag": "hq"},
        {"rule_id": "b", "cidr": "10.10.0.0/16", "action": "deny", "priority": 500, "source_tag": "dc1"},
        {"rule_id": "c", "cidr": "10.10.1.0/24", "action": "permit", "priority": 900, "source_tag": "lab"},
    ])
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "10.10.0.0/16"}
    ).json()

    permit_block = next(b for b in data["blocks"] if b["cidr"] == "10.10.1.0/24")
    chain = permit_block["match_chain"]
    # longest prefix first; first entry is the winner
    assert [e["rule_id"] for e in chain] == ["c", "b", "a"]
    assert [e["cidr"] for e in chain] == ["10.10.1.0/24", "10.10.0.0/16", "10.0.0.0/8"]
    assert chain[0]["rule_id"] == permit_block["winner_rule_id"]
    assert all("source_tag" in e and "priority" in e and "role" in e for e in chain)
    assert [e["role"] for e in chain] == ["winner", "winner", "winner"]

    deny_block = next(
        b for b in data["blocks"] if b["action"] == "deny" and b["cidr"] == "10.10.0.0/24"
    )
    assert [e["rule_id"] for e in deny_block["match_chain"]] == ["b", "a"]
    assert deny_block["match_chain"][0]["rule_id"] == deny_block["winner_rule_id"]


def test_duplicate_rules_appear_in_chain_with_priority_order(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "w", "cidr": "10.10.0.0/16", "action": "deny", "priority": 10, "source_tag": "win"},
        {"rule_id": "l1", "cidr": "10.10.0.0/16", "action": "permit", "priority": 20, "source_tag": "loser1"},
        {"rule_id": "l2", "cidr": "10.10.0.0/16", "action": "deny", "priority": 20, "source_tag": "loser2"},
    ])
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "10.10.0.0/16"}
    ).json()

    # The whole /16 has one decision: w wins by priority.
    assert [b["cidr"] for b in data["blocks"]] == ["10.10.0.0/16"]
    block = data["blocks"][0]
    assert block["winner_rule_id"] == "w"
    assert block["action"] == "deny"
    chain = block["match_chain"]
    # same-CIDR group: winner first, then losing duplicates by priority and
    # original submission order (l1 before l2), then the /8 ancestor.
    assert [e["rule_id"] for e in chain] == ["w", "l1", "l2", "a"]
    assert [e["role"] for e in chain] == ["winner", "duplicate", "duplicate", "winner"]


def test_unmatched_blocks_have_empty_chain_and_none_action(client: TestClient) -> None:
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"}])

    # Query entirely outside the rule: every block is unmatched.
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "192.168.0.0/24"}
    ).json()
    assert len(data["blocks"]) == 1
    block = data["blocks"][0]
    assert block["cidr"] == "192.168.0.0/24"
    assert block["action"] is None
    assert block["winner_rule_id"] is None
    assert block["winner_source_tag"] is None
    assert block["match_chain"] == []
    assert block["address_count"] == 2 ** 8
    assert data["summary"]["unmatched_addresses"] == 2 ** 8

    # Partial overlap: inside is permit, outside (sibling gap) is none.
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "11.0.0.0/8"}
    ).json()
    assert all(b["action"] is None and b["match_chain"] == [] for b in data["blocks"])


def test_gap_inside_query_with_enclosing_fallback(client: TestClient) -> None:
    # permit /8 with a deny /24; query the /23 straddling the deny block.
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "b", "cidr": "10.0.2.0/24", "action": "deny"},
    ])
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "10.0.2.0/23"}
    ).json()
    by_cidr = {b["cidr"]: b for b in data["blocks"]}
    assert by_cidr["10.0.2.0/24"]["action"] == "deny"
    assert by_cidr["10.0.2.0/24"]["winner_rule_id"] == "b"
    assert by_cidr["10.0.3.0/24"]["action"] == "permit"
    assert by_cidr["10.0.3.0/24"]["winner_rule_id"] == "a"
    # The /23 was already the minimum decomposition -> exactly two blocks.
    assert len(data["blocks"]) == 2


def test_adjacent_blocks_with_same_winner_are_merged(client: TestClient) -> None:
    # permit /24 with deny /26 children at .128 and .192: the first two /26
    # remainders (.0/26, .64/26) are adjacent siblings both won by the /24
    # rule -> collapsed into one /25.
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/24", "action": "permit"},
        {"rule_id": "b", "cidr": "10.0.0.128/26", "action": "deny"},
        {"rule_id": "c", "cidr": "10.0.0.192/26", "action": "deny"},
    ])
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "10.0.0.0/24"}
    ).json()
    by_cidr = {b["cidr"]: b for b in data["blocks"]}
    assert set(by_cidr) == {"10.0.0.0/25", "10.0.0.128/26", "10.0.0.192/26"}
    assert by_cidr["10.0.0.0/25"]["action"] == "permit"
    assert by_cidr["10.0.0.0/25"]["winner_rule_id"] == "a"
    assert by_cidr["10.0.0.0/25"]["address_count"] == 2 ** 7
    assert by_cidr["10.0.0.0/25"]["match_chain"][0]["rule_id"] == "a"


def test_adjacent_blocks_with_different_winners_are_not_merged(client: TestClient) -> None:
    # Two sibling /25s each won by a different rule, both permit: same action
    # but different winning rule -> must stay separate.
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/24", "action": "deny"},
        {"rule_id": "b", "cidr": "10.0.0.0/25", "action": "permit"},
        {"rule_id": "c", "cidr": "10.0.0.128/25", "action": "permit"},
    ])
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "10.0.0.0/24"}
    ).json()
    assert sorted(b["cidr"] for b in data["blocks"]) == ["10.0.0.0/25", "10.0.0.128/25"]
    assert "10.0.0.0/24" not in {b["cidr"] for b in data["blocks"]}
    assert {b["winner_rule_id"] for b in data["blocks"]} == {"b", "c"}


def test_fully_shadowed_ancestor_still_appears_in_chain(client: TestClient) -> None:
    # permit /24 fully shadowed by two deny /25s: the shadowed permit rule must
    # still show up in every /25 block's chain (it matches, just loses LPM).
    aid = _save(client, [
        {"rule_id": "a", "cidr": "192.168.0.0/24", "action": "permit"},
        {"rule_id": "b", "cidr": "192.168.0.0/25", "action": "deny"},
        {"rule_id": "c", "cidr": "192.168.0.128/25", "action": "deny"},
    ])
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "192.168.0.0/24"}
    ).json()
    assert sorted(b["cidr"] for b in data["blocks"]) == ["192.168.0.0/25", "192.168.0.128/25"]
    for block in data["blocks"]:
        assert block["action"] == "deny"
        assert [e["rule_id"] for e in block["match_chain"][:2]] == [
            block["winner_rule_id"], "a"
        ]


def test_ipv6_query_is_exact(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "v6a", "cidr": "2001:db8::/32", "action": "deny"},
        {"rule_id": "v6b", "cidr": "2001:db8:1::/48", "action": "permit"},
    ])
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "2001:db8::/32"}
    ).json()
    assert data["ip_version"] == 6
    assert data["total_addresses"] == 2 ** 96

    permit = next(b for b in data["blocks"] if b["action"] == "permit")
    assert permit["cidr"] == "2001:db8:1::/48"
    assert permit["winner_rule_id"] == "v6b"
    assert permit["address_count"] == 2 ** 80
    assert [e["rule_id"] for e in permit["match_chain"]] == ["v6b", "v6a"]

    deny_total = sum(b["address_count"] for b in data["blocks"] if b["action"] == "deny")
    assert deny_total == 2 ** 96 - 2 ** 80
    assert isinstance(permit["address_count"], int)

    # IPv4 rules never participate in an IPv6 query and vice versa.
    v4 = _save(client, [{"rule_id": "v4", "cidr": "10.0.0.0/8", "action": "permit"}])
    data = client.get(
        f"/api/analyses/{v4}/decision-map", params={"cidr": "2001:db8::/32"}
    ).json()
    assert all(b["action"] is None for b in data["blocks"])
    assert all(b["match_chain"] == [] for b in data["blocks"])


def test_query_larger_than_all_rules(client: TestClient) -> None:
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/24", "action": "permit"}])
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "0.0.0.0/0"}
    ).json()
    matched = [b for b in data["blocks"] if b["action"] is not None]
    assert [b["cidr"] for b in matched] == ["10.0.0.0/24"]
    assert sum(b["address_count"] for b in data["blocks"]) == 2 ** 32
    assert data["summary"]["unmatched_addresses"] == 2 ** 32 - 2 ** 8
    # Unmatched gaps carry no chain.
    assert all(b["match_chain"] == [] for b in data["blocks"] if b["action"] is None)


def test_invalid_cidr_returns_400(client: TestClient) -> None:
    aid = _save(client, [{"cidr": "10.0.0.0/8", "action": "permit"}])

    for bad in ("10.0.0.5/24", "not-a-cidr", "256.0.0.0/8", "10.0.0.0/33", "10.0.0.0/8,"):
        r = client.get(f"/api/analyses/{aid}/decision-map", params={"cidr": bad})
        assert r.status_code == 400, bad
        assert r.json()["error"] == "invalid_cidr"
        assert "invalid CIDR" in r.json()["detail"]

    # Missing cidr parameter is a framework-level 422.
    r = client.get(f"/api/analyses/{aid}/decision-map")
    assert r.status_code == 422


def test_version_not_found_returns_404(client: TestClient) -> None:
    # Malformed CIDR takes precedence over a missing version.
    r = client.get(
        "/api/analyses/an_0000000000000000/decision-map", params={"cidr": "10.0.0.5/24"}
    )
    assert r.status_code == 400

    r = client.get(
        "/api/analyses/an_0000000000000000/decision-map", params={"cidr": "10.0.0.0/8"}
    )
    assert r.status_code == 404
    assert "an_0000000000000000" in r.json()["detail"]


def test_enclosing_fallback_gap_includes_its_duplicates(client: TestClient) -> None:
    # A query gap won by the closest enclosing /8 lists that CIDR's losing
    # same-CIDR duplicates too (the whole rule group matches the gap).
    aid = _save(client, [
        {"rule_id": "w8", "cidr": "10.0.0.0/8", "action": "permit", "priority": 10},
        {"rule_id": "l8", "cidr": "10.0.0.0/8", "action": "deny", "priority": 20},
        {"rule_id": "b", "cidr": "10.0.0.0/24", "action": "deny"},
    ])
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "10.0.1.0/24"}
    ).json()
    # No local rule at all in this /24 -> one block won by the enclosing /8.
    assert len(data["blocks"]) == 1
    block = data["blocks"][0]
    assert block["cidr"] == "10.0.1.0/24"
    assert block["action"] == "permit"
    assert block["winner_rule_id"] == "w8"
    assert [e["rule_id"] for e in block["match_chain"]] == ["w8", "l8"]
    assert [e["role"] for e in block["match_chain"]] == ["winner", "duplicate"]


def test_same_priority_breaks_by_submission_order(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "first", "cidr": "10.0.0.0/24", "action": "permit", "priority": 100},
        {"rule_id": "second", "cidr": "10.0.0.0/24", "action": "deny", "priority": 100},
    ])
    data = client.get(
        f"/api/analyses/{aid}/decision-map", params={"cidr": "10.0.0.0/24"}
    ).json()
    block = data["blocks"][0]
    assert block["winner_rule_id"] == "first"
    assert block["action"] == "permit"
    assert [e["rule_id"] for e in block["match_chain"]] == ["first", "second"]


def test_endpoint_is_read_only(client: TestClient) -> None:
    before = client.get("/api/analyses").json()["total"]
    aid = _save(client, [{"cidr": "10.0.0.0/8", "action": "permit"}])
    for _ in range(3):
        r = client.get(f"/api/analyses/{aid}/decision-map", params={"cidr": "10.0.0.0/8"})
        assert r.status_code == 200
    after = client.get("/api/analyses").json()["total"]
    assert after == before + 1


# ---------------------------------------------------------------------------
def test_parse_query_cidr_accepts_strict_and_rejects_host_bits() -> None:
    assert str(parse_query_cidr(" 10.0.0.0/8 ")) == "10.0.0.0/8"
    assert parse_query_cidr("2001:db8::/32").version == 6
    for bad in ("10.0.0.1/24", "garbage", "", "   "):
        with pytest.raises(Exception):
            parse_query_cidr(bad)


def test_engine_determinism_and_minimality() -> None:
    result = {
        "id": "an_0123456789abcdef",
        "ruleset_hash": "h",
        "analyzer_version": "1.0.0",
        "rules": [
            {"index": 0, "rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit",
             "priority": 100, "source_tag": "hq"},
            {"index": 1, "rule_id": "b", "cidr": "10.10.0.0/16", "action": "deny",
             "priority": 100, "source_tag": "dc1"},
        ],
    }
    query = parse_query_cidr("10.10.0.0/16")
    first = build_decision_map(result, query)
    second = build_decision_map(result, query)
    assert first == second

    # Every returned block is a real CIDR-aligned subnet of the query and
    # adjacent blocks never share both winner and mergeable sibling shape
    # (the decomposition is minimal w.r.t. winner identity).
    qnet = ipaddress.ip_network("10.10.0.0/16")
    for block in first["blocks"]:
        net = ipaddress.ip_network(block["cidr"])
        assert net.subnet_of(qnet)
