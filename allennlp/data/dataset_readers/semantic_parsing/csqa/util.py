
def get_dummy_action_sequences(question_entities):
    return [['@start@ -> Set[Entity]', 'Set[Entity] -> [<Entity:Set[Entity]>, Entity]', '<Entity:Set[Entity]> -> get',
             'Entity -> ' + question_entities[-1]]]


def question_is_indirect(question_description, question_type):
    return "Indirect" in question_description or "indirect" in question_description or "indirectly" in \
           question_description or "Incomplete" in question_description or "Coreferenced" in question_type or \
           "Ellipsis" in question_type
