import torch

TORCH_DTYPES = {
    'no': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.float16,
}

def exists(val):
    return val is not None

# Function to convert a verb from root form to third person singular form
def convert_to_third_person_singular(verb):
    # Define a list of common irregular verbs and their third person singular forms
    irregular_verbs = {
        "be": "is",
        "have": "has",
        "do": "does",
        # Add more irregular verbs as needed
    }

    if verb in irregular_verbs:
        return irregular_verbs[verb]
    else:
        if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
            return verb[:-1] + "ies"
        elif verb.endswith(("o", "s", "x", "z")) or verb.endswith(("sh", "ch")):
            return verb + "es"
        else:
            return verb + "s"

def cos_sim(a, b, eps=1e-8):
    if len(a.shape) < 2:
        a = torch.unsqueeze(a, 0)
        b = torch.unsqueeze(b, 0)
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt