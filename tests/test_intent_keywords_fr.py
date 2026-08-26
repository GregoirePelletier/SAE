"""Teste src/data/dataset.py::INTENT_KEYWORDS_FR -- correctif B.6
(AUDIT_SAE_2026-08.md) : le motif "remboursement" matchait "avoir\\w*", le
verbe "avoir" (l'un des mots les plus fréquents du français), invalidant tout
document qui le contient comme faussement étiqueté "remboursement". Le motif
corrigé exige un déterminant devant "avoir" (le nom, note de crédit EDF)
plutôt que la forme verbale nue.

N5 (AUDIT_SAE_2026-08.md §8) : "l'avoir"/"d'avoir" retirés du motif après
vérification empirique sur `local_data/emails/Mails.tsv` -- 31/31 occurrences
de "d'avoir" et 4/4 de "l'avoir" y sont l'usage VERBAL (infinitif après
préposition/pronom COD, ex. "je ne suis pas certain d'avoir compris", "je
vous remercie de me l'avoir envoyée"), 0 nominal. "l'avoir" est
grammaticalement ambigu (article défini + nom "le avoir" élidé, OU pronom COD
"l'" + infinitif "avoir") et n'est pas fiablement désambiguïsable par regex ;
seuls "un/mon/notre avoir" (déterminant qui ne peut précéder qu'un nom) sont
fiables."""
import re

from src.data.dataset import INTENT_KEYWORDS_FR

PAT_REMBOURSEMENT = INTENT_KEYWORDS_FR["remboursement"]


def _matches(text):
    return bool(re.search(PAT_REMBOURSEMENT, text, flags=re.I))


def test_verb_avoir_no_longer_matches():
    assert not _matches("Je souhaite avoir des informations sur mon contrat.")
    assert not _matches("Pourriez-vous avoir l'obligeance de me répondre ?")


def test_ambiguous_l_avoir_d_avoir_no_longer_matches():
    # N5 : formes ambiguës retirées -- 100% verbal dans le corpus réel testé,
    # aucun cas nominal observé (cf. docstring du module).
    assert not _matches("Je ne suis pas certain d'avoir compris toutes les clauses.")
    assert not _matches("Je vous remercie de me l'avoir envoyée.")
    assert not _matches("Merci de m'envoyer l'avoir correspondant.")


def test_noun_avoir_credit_note_still_matches():
    assert _matches("Vous me devez un avoir sur ma prochaine facture.")
    assert _matches("Je vous demande mon avoir sur la prochaine facture.")
    assert _matches("Merci de créditer notre avoir sur le compte.")


def test_core_reimbursement_terms_still_match():
    assert _matches("Je demande le remboursement du trop-perçu.")
    assert _matches("Il y a eu un trop perçu sur ma facture.")


def test_other_intents_unaffected():
    assert re.search(INTENT_KEYWORDS_FR["urgence"], "Coupure urgente à réparer.", flags=re.I)
    assert re.search(INTENT_KEYWORDS_FR["reclamation"], "Je conteste cette réclamation.", flags=re.I)
