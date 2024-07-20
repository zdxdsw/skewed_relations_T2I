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
