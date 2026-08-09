from scripts.augmentation_lexical_leakage_audit import stock_phrase_templating_rate


def test_stock_phrase_templating_rate_detects_repeated_phrase():
    texts_by_class = {
        "templated": ["je vous remercie pour votre aide"] * 8 + ["merci beaucoup pour tout"] * 2,
        "diverse": [
            "le chat mange une pomme rouge",
            "les nuages flottent doucement dans le ciel bleu",
            "un train rapide traverse la campagne verte",
            "elle chante une chanson joyeuse ce matin",
            "nous marchons vers la montagne enneigée",
            "il pleut fort sur la ville endormie",
            "des oiseaux volent haut au-dessus des arbres",
            "la rivière coule vers un lac paisible",
            "un vent frais souffle sur la plage déserte",
            "le soleil brille sur les champs dorés",
        ],
    }
    rates = stock_phrase_templating_rate(texts_by_class)
    assert rates["templated"] > rates["diverse"]
    assert rates["templated"] >= 0.8
