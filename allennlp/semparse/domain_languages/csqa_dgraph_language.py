from allennlp.semparse.contexts import CSQADgraphContext
from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, ExecutionError, predicate)

import logging
import json

from typing import Dict, List, NamedTuple, Tuple, Union
from numbers import Number

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Predicate(NamedTuple):
    name: str
    id: Union[str, int]


class Entity(NamedTuple):
    name: str
    id: Union[str, int]


class CSQADgraphLanguage(DomainLanguage):
    # pylint: disable=too-many-public-methods,no-self-use
    """
    Implements the functions in the variable free language in "Dialog-to-Action: Conversational Question
    Answering Over a Large-Scale Knowledge Base" by Daya Guo, Duyu Tang, Nan Duan, Ming Zhou, and Jian Yin
    """
    def __init__(self,
                 csqa_dgraph_context: CSQADgraphContext
                 ) -> None:
        # TODO: do we need dates here too?
        # TODO: check name and value passed to add_constant
        super().__init__(start_types={Number, List[Entity]})
        self.kg_context = csqa_dgraph_context
        self.kg_data = csqa_dgraph_context.kg_data
        self.use_integer_ids = False

        for predicate_id, predicate in csqa_dgraph_context.predicate_id2string.items():
            self.add_constant(predicate_id, self.get_predicate_from_question_id(predicate_id), type_=Predicate)
            # add inverse
            predicate_id = predicate_id[0] + "-" + predicate_id[1:]
            self.add_constant(predicate_id, self.get_predicate_from_question_id(predicate_id), type_=Predicate)

        # add fake id for istypeof / isoftype
        predicate_id = "P1"
        self.add_constant(predicate_id, self.get_predicate_from_question_id(predicate_id), type_=Predicate)

        predicate_id = "P-1"
        self.add_constant(predicate_id, self.get_predicate_from_question_id(predicate_id), type_=Predicate)

        question_entities, question_numbers = csqa_dgraph_context.get_entities_from_question()

        self._question_entities = question_entities

        for entity_id in self._question_entities:  # + csqa_dgraph_context.question_type_list:
            self.add_constant(entity_id, self.get_entity_from_question_id(entity_id), type_=Entity)

        self._question_numbers = [number for number, _ in question_numbers]

        for number in self._question_numbers:
            self.add_constant(str(number), float(number), type_=Number)

        # TODO: remove this
        self.add_constant(str(0), float(0), type_=Number)

        # Mapping from terminal strings to productions that produce them.  We use this in the
        # agenda-related methods, and some models that use this language look at this field to know
        # how many terminals to plan for.
        self.terminal_productions: Dict[str, str] = {}
        for name, types in self._function_types.items():
            # if "P" is not name[0]:
            #     print("%s -> %s" % (types[0], name))
            self.terminal_productions[name] = "%s -> %s" % (types[0], name)

    def get_agenda(self):
        # TODO: this needs to be implemented when carrying out a search for correct logical forms
        raise NotImplementedError("")

    def get_entity_from_question_id(self, entity_id: str):
        # these are always strings with Q prefix
        if self.use_integer_ids:
            return Entity(entity_id, int(entity_id[1:]))
        else:
            return Entity(entity_id, entity_id)

    def get_entity_from_kg_id(self, entity_id: Union[str, int]):
        # This is exactly the inverse of get_entity_from_question_id,
        # we get an id, which can be without Q prefix or with, and we want to give it the right name
        if self.use_integer_ids:
            # entity id is an int
            return Entity("Q" + str(entity_id), entity_id)
        else:
            return Entity(entity_id, entity_id)

    def get_predicate_from_question_id(self, predicate_id: str):
        # these are always strings with P prefix
        if self.use_integer_ids:
            return Predicate(predicate_id, int(predicate_id[1:]))
        else:
            return Predicate(predicate_id, predicate_id)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.kg_data == other.kg_data and self.terminal_productions == other.terminal_productions
        return NotImplemented

    def evaluate_logical_form_correct(self, logical_form: str, target_list: List[str]) -> bool:
        """
        Takes a logical form, and the list of target entities as strings from the original lisp
        string, and returns True iff the logical form executes to the entity list
        """
        assert len(target_list) == 1
        try:
            denotation = self.execute(logical_form)
        except ExecutionError:
            logger.warning(f'Failed to execute: {logical_form}')
            return False

        return set(denotation) == set(target_list)

    def evaluate_logical_form_precision_recall(self, logical_form: str, target_list: List[str]) -> Tuple[float, float]:
        """
        Takes a logical form, and the list of target entities as strings from the original lisp
        string, and returns precision and recall
        """
        try:
            denotation = self.execute(logical_form)
        except ExecutionError:
            logger.warning(f'Failed to execute: {logical_form}')
            return 0., 0.

        n_intersection = len(set(denotation).intersection(set(target_list)))
        precision = n_intersection / len(set(denotation))
        recall = n_intersection / len(set(target_list))

        return precision, recall

    @predicate
    def find(self, entities: List[Entity], predicate_: Predicate) -> List[Entity]:
        """
        find function takes a list of entities E and and a predicate p and loops through
        e in E and returns the set of entities with a p edge to e
        """

        """Get the property of a list of entities."""
        result = []
        for ent in entities:
            if not(predicate_.id == 1 or predicate_.id == -1):
                query = """query all($wikidata_uid: string) {
                  q(func: eq(wikidata_uid, $wikidata_uid))
                  {
                    """ + predicate_.id + """ {
                      wikidata_uid
                    }
                  }
                }"""
                variables = {'$wikidata_uid': ent.name}
                res = self.kg_data.query(query, variables=variables)
                res_json = json.loads(res.json)
                ent_ids = []
                for obj in res_json['q'][0][predicate_.name]:
                    ent_ids.append(obj['wikidata_uid'])
                for ent_id in ent_ids:
                    entity = self.get_entity_from_kg_id(ent_id)
                    result.append(entity)
                    # result = result.union({entity})
            else:
                try:
                    ent_ids: List[Union[str, int]] = self.kg_type_data[ent.id][predicate_.id]
                    for ent_id in ent_ids:
                        entity = self.get_entity_from_kg_id(ent_id)
                        result.append(entity)
                        # result = result.union({entity})
                except KeyError:
                    continue
        return list(set(result))

    @predicate
    def count(self, entities: List[Entity]) -> Number:
        """
        returns a count of the passed list of entities
        """
        return len(entities)  # type: ignore

    @predicate
    def is_in(self, entity: Entity, entities: List[Entity]) -> bool:
        """
        return whether the first entity is in the set of entities

        """
        return entity in entities

    @predicate
    def union(self, entities1: List[Entity], entities2: List[Entity]) -> List[Entity]:
        """
        return union of two sets of entities

        """
        return list(set(entities1).union(set(entities2)))

    @predicate
    def intersection(self, entities1: List[Entity], entities2: List[Entity]) -> List[Entity]:
        """
        return intersection of two sets of entities

        """
        return list(set(entities1).intersection(entities2))

    @predicate
    def get(self, entity: Entity)-> List[Entity]:
        """
        get entity and wrap it in a set (See Dialog-to-action Table 1 A15)

        """
        return [entity]

    @predicate
    def diff(self, entities1: List[Entity], entities2: List[Entity])-> List[Entity]:
        """
        return instances included in entities1 but not included in entities2. Note that this is *NOT* the symmetric
        difference. E.g. set([1, 2]) - set([2, 3]) = set([1]) and *NOT* set([1, 2]) - set([2, 3]) = set([1, 3])

        """
        return list(set(entities1) - set(entities2))

    @predicate
    def larger(self, entities: List[Entity], predicate_: Predicate, num: Number)-> List[Entity]:
        """
        subset of entities linking to more than num entities with predicate_
        """

        result = set()
        for entity in entities:
            if len(self.find([entity], predicate_)) > num:
                result = result.union([entity])

        return list(result)

    @predicate
    def less(self, entities: List[Entity], predicate_: Predicate, num: Number)-> List[Entity]:
        """
        subset of entities linking to less than num entities with predicate_
        """
        # TODO (koen): do we want to include entities that have 0 relations?
        result = []
        for entity in entities:
            n_links = len(self.find([entity], predicate_))
            if n_links < num and n_links != 0:  # if n_links < num:
                result.append(entity)

        return list(set(result))

    @predicate
    def equal(self, entities: List[Entity], predicate_: Predicate, num: Number)-> List[Entity]:
        """
        subset of entities linking to exactly num entities with predicate_
        """

        result = set([entity for entity in entities if len(self.find([entity], predicate_)) == num])
        return list(result)

    @predicate
    def most(self, entities: List[Entity], predicate_: Predicate, num: Number)-> List[Entity]:
        """
        subset of entities linking to at most num entities with predicate_
        """

        result = []
        for entity in entities:
            n_links = len(self.find([entity], predicate_))
            if n_links <= num and n_links != 0:
                result.append(entity)

        return list(set(result))

    @predicate
    def least(self, entities: List[Entity], predicate_: Predicate, num: Number)-> List[Entity]:
        """
        subset of entities linking to at least num entities with predicate_
        """

        result = set()
        for entity in entities:
            if len(self.find([entity], predicate_)) >= num:
                result = result.union([entity])

        return list(result)
