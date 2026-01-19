def replay_after_restart(load_router):
    router = load_router()   # loads persisted SQLite state

    stats = router.stats()
    beliefs = router.world_model.stats()

    assert stats, "Bandit stats missing after restart"
    assert beliefs, "World model lost memory"

    return {
        "bandit_stats": stats,
        "beliefs": beliefs
    }
