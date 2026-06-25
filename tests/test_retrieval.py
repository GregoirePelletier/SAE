import torch

def test_cosine_retrieval():

    docs = torch.randn(100, 512)

    query = docs[0]

    scores = torch.nn.functional.cosine_similarity(
        docs,
        query.unsqueeze(0),
        dim=1
    )

    idx = scores.argmax()

    assert idx == 0