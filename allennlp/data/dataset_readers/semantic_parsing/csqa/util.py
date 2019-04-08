from pathlib import Path
from typing import List
import numpy as np
from allennlp.data.tokenizers.token import Token

from allennlp.data.fields import ArrayField


def get_dummy_action_sequences(question_entities, question_type_entities):
    if question_entities:
        return [['@start@ -> Set[Entity]', 'Set[Entity] -> [<Entity:Set[Entity]>, Entity]',
                 '<Entity:Set[Entity]> -> get', 'Entity -> ' + question_entities[-1]]]
    elif question_type_entities:
        return [['@start@ -> Set[Entity]', 'Set[Entity] -> [<Entity:Set[Entity]>, Entity]',
                 '<Entity:Set[Entity]> -> get', 'Entity -> ' + question_type_entities[-1]]]
    else:
        raise ValueError()


def question_is_indirect(question_description, question_type):
    return "Indirect" in question_description or "indirect" in question_description or "indirectly" in \
           question_description or "Incomplete" in question_description or "Coreferenced" in question_type or \
           "Ellipsis" in question_type


def parse_answer(answer, entities_result, language):
    if answer in ["YES", "NO"]:
        return True if answer is "YES" else False
    elif entities_result:
        # Check if result is a count of entities.
        try:
            return int(answer)
        # Read entities in result.
        except ValueError:
            return {language.get_entity_from_question_id(ent) for ent in entities_result}
    elif answer.startswith("Did you mean"):
        return "clarification"
    elif answer == "YES and NO respectively" or answer == "NO and YES respectively":
        return None
    else:
        raise ValueError("unknown answer format: {}".format(answer))


def maybe_add_entities(question,
                       question_entities,
                       question_type_entities,
                       context,
                       add_entities_to_sentence,
                       separator=" [SEP] "):
    """
    maybe add entities to question to enable BERT to attend entities
    """
    entity_id2string = context.entity_id2string
    if add_entities_to_sentence:
        all_q_entities_string = [entity_id2string[ent] for ent in question_entities + question_type_entities]
        if all_q_entities_string:
            question += separator + separator.join(all_q_entities_string)
    return question


def get_extraction_dir(path: Path):
    if path.match('*.tar.gz'):
        # remove .tar.gz extension
        return path.parent / path.stem.split('.')[0]
    else:
        # add _extracted to cached path
        return Path(str(path) + "_extracted")


def get_segment_field_from_tokens(tokenized_question: List[Token], sep="[SEP]"):
    tokenized_question_strings = [t.text for t in tokenized_question]
    if sep not in tokenized_question_strings:
        return ArrayField(np.array([0] * len(tokenized_question)))
    else:
        n_zero_segments = tokenized_question_strings.index(sep) + 1
        n_one_segments = len(tokenized_question_strings) - n_zero_segments
        segment_ids = [0] * n_zero_segments + [1] * n_one_segments
        return ArrayField(np.array(segment_ids))


def prepare_question_for_bert(question,
                              question_entities,
                              question_type_entities,
                              context,
                              add_entities_to_sentence):

    question = "[CLS] " + question
    question = maybe_add_entities(question, question_entities, question_type_entities, context,
                                  add_entities_to_sentence)
    return question


