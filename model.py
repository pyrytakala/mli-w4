
# image embeddings:
#   use all but last layer from CLIP (see clip_emb_demo.py)
#   add one linear projection layer 768 -> 512 (the text embeddings are 512 in dimension)

# text embeddings:
#   use embeddings from CLIP's text model (see clip_emb_demo.py)

# decoder logic:
#     if image patch, encode with image encoder (see clip_emb_demo.py)
#     else, encode with text encoder
#     pass through decoder



