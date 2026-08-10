from run_config.make_partitioned_gpu_funnels import (
    FINAL_STRICT_HOURS,
    LOCAL_DATABASE,
    LOCAL_UNDEREXPLORED_CAPS,
    RESOLUTIONS,
    SERVER_UNDEREXPLORED_CAPS,
    STRICT_ENDPOINTS,
    TRACK_A_DATABASE,
    TRACK_B_DATABASE,
    TWO_HOURS,
    UNDEREXPLORED_ENDPOINTS,
    local_track_documents,
    track_a_documents,
    track_b_documents,
)


def _by_name(documents):
    return {document.name: document for document in documents}


def test_track_b_is_stage_major_and_reduces_every_endpoint_population():
    documents = track_b_documents()
    lane_count = len(SERVER_UNDEREXPLORED_CAPS) * 2

    assert len(documents) == lane_count * 7
    assert all(document.runtime.database == TRACK_B_DATABASE for document in documents)
    assert all("scout500" in document.name for document in documents[:lane_count])
    assert all(
        "broad500" in document.name
        for document in documents[lane_count : 2 * lane_count]
    )
    assert all(
        "deep250" in document.name
        for document in documents[2 * lane_count : 3 * lane_count]
    )
    assert all(
        "polish100" in document.name
        for document in documents[3 * lane_count : 4 * lane_count]
    )
    assert all(
        "loose10" in document.name
        for document in documents[4 * lane_count : 5 * lane_count]
    )
    assert all(
        "strict3" in document.name
        for document in documents[5 * lane_count : 6 * lane_count]
    )
    assert all("final_strict1" in document.name for document in documents[-lane_count:])


def test_every_broad_seed_reaches_ten_thousand_steps_before_reduction():
    documents = track_b_documents()
    lane_count = len(SERVER_UNDEREXPLORED_CAPS) * 2

    for scout, broad in zip(
        documents[:lane_count], documents[lane_count : 2 * lane_count]
    ):
        assert scout.runtime.initialisations == 500
        assert scout.scalar_cases()[0].schedule == ((2_000, 1.0),)
        assert broad.scalar_cases()[0].schedule == ((8_000, 0.5),)
        assert broad.query.where["config_id"] == scout.config_id
        assert broad.query.limit is None
        assert scout.runtime.max_elapsed_seconds is None
        assert broad.runtime.max_elapsed_seconds is None


def test_refinement_levels_have_two_hour_guards_and_exact_dependencies():
    documents = track_b_documents()
    named = _by_name(documents)

    for document in documents:
        if "scout500" in document.name or "broad500" in document.name:
            continue
        assert document.runtime.max_elapsed_seconds == TWO_HOURS
        assert document.query is not None
        assert document.query.where["config_id"] in {
            candidate.config_id for candidate in documents
        }
        source = next(
            candidate
            for candidate in documents
            if candidate.config_id == document.query.where["config_id"]
        )
        assert named[source.name].config_id == source.config_id


def test_track_a_covers_both_endpoints_at_every_grid_before_final_strict_pass():
    documents = track_a_documents()
    names = {document.name for document in documents}

    for N in RESOLUTIONS:
        for cap in STRICT_ENDPOINTS:
            for endpoint in ("low", "high"):
                preliminary = f"N{N}_u{cap}_{endpoint}_strictgrid_v1_strict3_gpu"
                final = f"N{N}_u{cap}_{endpoint}_strictgrid_v1_final_strict1_gpu"
                assert preliminary in names
                assert final in names
                final_document = next(
                    document for document in documents if document.name == final
                )
                assert final_document.runtime.max_elapsed_seconds == FINAL_STRICT_HOURS
                assert final_document.query.where["config_name"] == preliminary
                assert final_document.scalar_cases()[0].J_tol == 1e-6
                assert final_document.scalar_cases()[0].projected_gradient_tol == 1e-5

    assert all(document.runtime.database == TRACK_A_DATABASE for document in documents)


def test_higher_grid_promotions_keep_three_predecessors_and_add_twenty_starts():
    documents = track_a_documents()

    promoted = [document for document in documents if "_loose23_gpu" in document.name]
    assert len(promoted) == (len(RESOLUTIONS) - 1) * len(STRICT_ENDPOINTS) * 2
    for document in promoted:
        assert document.runtime.initialisations == 20
        assert document.query.limit == 3
        assert document.runtime.max_elapsed_seconds == TWO_HOURS
        assert document.scalar_cases()[0].optimizer == "lbfgs"


def test_u1280_low_reuses_the_completed_thousand_seed_source():
    first = track_a_documents()[0]

    assert first.name == "N100_u1280_low_strictgrid_v1_strict3_gpu"
    assert first.query.database == "results/bar_endpoint_seed1000_loose_u320.sqlite3"
    assert first.query.limit == 3
    assert first.query.where["config_name"].endswith("seed1000_top10_loose_gpu")


def test_server_and_laptop_partition_every_underexplored_cap_once():
    assert set(SERVER_UNDEREXPLORED_CAPS).isdisjoint(LOCAL_UNDEREXPLORED_CAPS)
    assert set(SERVER_UNDEREXPLORED_CAPS) | set(LOCAL_UNDEREXPLORED_CAPS) == set(
        UNDEREXPLORED_ENDPOINTS
    )

    local = local_track_documents()
    lane_count = len(LOCAL_UNDEREXPLORED_CAPS) * 2
    assert len(local) == lane_count * 7
    assert all(document.runtime.database == LOCAL_DATABASE for document in local)
    assert all(
        document.scalar_cases()[0].u_max in LOCAL_UNDEREXPLORED_CAPS
        for document in local
    )
    assert all(document.runtime.device == "gpu" for document in local)
    assert all(
        document.runtime.max_initialisations_per_batch <= 50
        for document in local
    )
