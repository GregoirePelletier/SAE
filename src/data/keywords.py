ENERGY_KEYWORDS = [
    "énergie", "électrique", "thermique", "nucléaire", "hydraulique",
    "réseau électrique", "turbine", "réacteur", "transformateur",
    "puissance", "tension électrique", "courant électrique",
    "centrale électrique", "ligne haute tension", "barrage",
    "photovoltaïque", "éolien", "cogénération", "sous-station",
    "ingénierie", "système de contrôle", "maintenance industrielle",
    "capteur", "asservissement", "automate programmable", "SCADA",
    "instrumentation", "mécanique des fluides", "thermodynamique",
    "génie civil", "béton armé", "soudure", "corrosion",
    "sécurité industrielle", "habilitation électrique", "risque électrique",
    "compteur", "disjoncteur", "alternateur", "condensateur",
    "watt", "kilowatt", "mégawatt", "ampère", "volt",
    "EDF", "RTE", "Enedis", "CEA", "ASN", "IRSN",
    "fission", "fusion nucléaire", "combustible nucléaire", "uranium",
    "plutonium", "radioactivité", "déchets radioactifs", "démantèlement",
]

SPORTS_KEYWORDS = [
    "football", "tennis", "rugby", "cyclisme", "athlétisme",
    "natation", "basketball", "handball", "volleyball",
    "Tour de France", "championnat", "ligue", "compétition",
    "entraînement", "match", "tournoi", "Jeux olympiques",
    "équipe nationale", "coupe du monde", "stade", "arbitre",
    "marathon", "sprint", "relais", "médaille", "podium",
    "division", "transfert", "buteur", "gardien", "milieu de terrain",
    "entraîneur", "sélectionneur", "FIFA", "UEFA",
    "Roland-Garros", "Ligue 1", "Ligue des champions", "Grand Prix",
]

SUPPORT_KEYWORDS = [
    "urgence", "panne", "bloqué", "facture", "inadmissible", "honteux",
    "scandale", "immédiatement", "remboursement", "coupure", "vol",
    "cordialement", "bonjour", "serait-il possible", "raccordement",
    "mise en service", "déménagement", "souscription", "veuillez agréer",
    "inacceptable", "abusif", "arnaque", "véroles", "débile",
    "catastrophe", "pathétique", "déçu", "révolte", "rage",
    "réclamation", "litige", "contentieux", "tribunal", "avocat",
    "justice", "action en justice", "résolution des litiges",
]

ENERGY_URL_PATTERNS = [
    "edf.fr", "rte-france.com", "enedis.fr", "asn.fr",
    "irsn.fr", "cea.fr", "sfen.org", "connaissancedesenergies.org",
    "techniques-ingenieur.fr",
]

SPORTS_URL_PATTERNS = [
    "lequipe.fr", "eurosport.fr", "france-football.fr",
    "rugbyrama.fr", "sport.fr", "sofoot.com", "rmc.bfmtv.com/sport",
]

SUPPORT_URL_PATTERNS = [
    "reddit.com", "forum", "jeuxvideo.com", "hardware.fr",
    "60millions-mag", "quechoisir.org", "trustpilot.com", "avis-verifies",
    "consoglobe.com", "economiesolidaire.org",
]

DOMAIN_KEYWORDS_MAP = {
    "energy": ENERGY_KEYWORDS,
    "sports": SPORTS_KEYWORDS,
    "support": SUPPORT_KEYWORDS,
}

DOMAIN_URL_MAP = {
    "energy": ENERGY_URL_PATTERNS,
    "sports": SPORTS_URL_PATTERNS,
    "support": SUPPORT_URL_PATTERNS,
}
