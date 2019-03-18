from allennlp.semparse.contexts import CSQAContext
from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, ExecutionError,
                                                                predicate)
import logging

import time


from typing import Dict, List, NamedTuple, Set, Tuple, Union, Type
from numbers import Number
from collections import Counter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Predicate(NamedTuple):
    name: str
    id: Union[str, int]


class Entity(NamedTuple):
    name: str
    id: Union[str, int]


class TypeEntity(NamedTuple):
    name: str
    id: Union[str, int]


class TypePredicate(NamedTuple):
    name: str
    id: Union[str, int]


class CSQALanguage(DomainLanguage):
    # pylint: disable=too-many-public-methods,no-self-use
    """
    Implements the functions in the variable free language in "Dialog-to-Action: Conversational Question
    Answering Over a Large-Scale Knowledge Base" by Daya Guo, Duyu Tang, Nan Duan, Ming Zhou, and Jian Yin
    """
    def __init__(self,
                 csqa_context: CSQAContext,
                 search_modus: bool = False
                 ) -> None:
        # TODO: do we need dates here too?
        # TODO: check name and value passed to add_constant
        super().__init__(start_types={Set[Entity], Number, bool})
        self.search_modus = search_modus
        self.kg_context = csqa_context
        self.kg_data = csqa_context.kg_data
        self.kg_type_data = csqa_context.kg_type_data
        self.use_integer_ids = csqa_context.use_integer_ids
        question_entities, question_numbers = csqa_context.get_entities_from_question()
        self._question_entities = question_entities
        self._question_numbers = [number for number, _ in question_numbers]
        self.function_cache = []
        self.result_cache = []

        for predicate_id in csqa_context.question_predicates:
            inv_predicate_id = predicate_id[0] + "-" + predicate_id[1:]
            self.add_constant(predicate_id, self.get_predicate_from_question_id(predicate_id), type_=Predicate)
            self.add_constant(inv_predicate_id, self.get_predicate_from_question_id(inv_predicate_id), type_=Predicate)

        # add fake id for is_type_of / is_of_type
        if csqa_context.question_type in ["Quantitative Reasoning (All)", "Comparative Reasoning (All)"]:
            type_predicate_ids = ["P1", "P-1"]
        else:
            type_predicate_ids = ["P1"]

        for type_predicate_id in type_predicate_ids:
            self.add_constant(type_predicate_id,
                              self.get_predicate_from_question_id(type_predicate_id, predicate_class=Predicate),
                              type_=Predicate)

        for entity_id in self._question_entities + csqa_context.question_type_list:
            self.add_constant(entity_id, self.get_entity_from_question_id(entity_id), type_=Entity)

        for type_id in csqa_context.question_type_list:
            self.add_constant(type_id, self.get_entity_from_question_id(type_id, entity_class=TypeEntity),
                              type_=TypeEntity)

        for number in self._question_numbers:
            self.add_constant(str(number), float(number), type_=Number)

        # Mapping from terminal strings to productions that produce them.  We use this in the
        # agenda-related methods, and some models that use this language look at this field to know
        # how many terminals to plan for.
        self.terminal_productions: Dict[str, str] = {}
        for name, types in self._function_types.items():
            # if "P" is not name[0]:
            #     print("%s -> %s" % (types[0], name))
            self.terminal_productions[name] = "%s -> %s" % (types[0], name)

    def set_search_modus(self):
        self.search_modus = True

    def get_agenda(self):
        # TODO: this needs to be implemented when carrying out a search for correct logical forms
        raise NotImplementedError("")

    def get_entity_from_question_id(self, entity_id: str, entity_class: Type = Entity):
        # these are always strings with Q prefix
        if self.use_integer_ids:
            return entity_class(entity_id, int(entity_id[1:]))
        else:
            return entity_class(entity_id, entity_id)

    def get_entity_from_kg_id(self, entity_id: Union[str, int], entity_class: Type = Entity):
        # This is exactly the inverse of get_entity_from_question_id,
        # we get an id, which can be without Q prefix or with, and we want to give it the right name
        if self.use_integer_ids:
            # entity id is an int
            return entity_class("Q" + str(entity_id), entity_id)
        else:
            return entity_class(entity_id, entity_id)

    def get_predicate_from_question_id(self, predicate_id: str,
                                       predicate_class: Type = Predicate):
        # these are always strings with P prefix
        if self.use_integer_ids:
            return predicate_class(predicate_id, int(predicate_id[1:]))
        else:
            return predicate_class(predicate_id, predicate_id)

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
    def find(self, entities: Set[Entity], predicate_: Predicate) -> Set[Entity]:
        """
        find function takes a list of entities E and and a predicate p and loops through
        e in E and returns the set of entities with a p edge to e
        """

        result = set()
        kg_data = self.kg_data if predicate_.id not in [1, -1, "1", "-1"] else self.kg_type_data

        if self.search_modus:
            try:
                result = self.result_cache[self.function_cache.index(["find", entities, predicate_])]
                return result
            except ValueError:
                pass

        for ent in entities:
            # if not(predicate_.id == 1 or predicate_.id == -1):
            try:
                ent_ids: List[Union[str, int]] = kg_data[ent.id][predicate_.id]
                if len(ent_ids) > 10000 and self.search_modus and (not self.kg_context.question_type in [
                    "Quantitative Reasoning (All)", "Comparative Reasoning (All)"]):
                    # TODO: THIS IS A VERY BAD SOLUTION, FIX
                    ent_ids = ent_ids[:13]

                for ent_id in ent_ids:
                    entity = self.get_entity_from_kg_id(ent_id)
                    result.add(entity)

            except KeyError:
                continue

        if self.search_modus and len(result) > 1000:
            self.function_cache.append(["find", entities, predicate_])
            self.result_cache.append(result)

        return result

    # @predicate
    # def filter_type(self, entities: List[Entity], object_type: TypeEntity) -> List[Entity]:
    #     """
    #     find function takes a list of entities E and and a predicate p and loops through
    #     e in E and returns the set of entities with a p edge to e
    #     """
    #
    #     """Get the property of a list of entities."""
    #     result = []
    #     predicate_id = 1 if self.use_integer_ids else "P1"
    #     for ent in entities:
    #         try:
    #             type_ids: List[Union[str, int]] = self.kg_type_data[ent.id][predicate_id]
    #             if object_type.id in type_ids:
    #                 result.append(ent)
    #         except KeyError:
    #             continue
    #     result = list(set(result))
    #     return result

    @predicate
    def has_relation_with(self, entities: Set[Entity], predicate_: Predicate, object_: Entity) -> Set[Entity]:
        """
        """

        kg_data = self.kg_data if predicate_.id not in [1, -1, "1", "-1"] else self.kg_type_data
        result = set()
        for ent in entities:
            try:
                object_ids: List[Union[str, int]] = kg_data[ent.id][predicate_.id]
                if object_.id in object_ids:
                    result.add(ent)
            except KeyError:
                continue
        return result

    @predicate
    def count(self, entities: Set[Entity]) -> Number:
        """
        returns a count of the passed list of entities
        """
        return len(entities)  # type: ignore

    @predicate
    def is_in(self, entity: Entity, entities: Set[Entity]) -> bool:
        """
        return whether the first entity is in the set of entities

        """
        return entity in entities

    @predicate
    def union(self, entities1: Set[Entity], entities2: Set[Entity]) -> Set[Entity]:
        """
        return union of two sets of entities

        """
        return entities1.union(entities2)

    @predicate
    def intersection(self, entities1: Set[Entity], entities2: Set[Entity]) -> Set[Entity]:
        """
        return intersection of two sets of entities

        """
        return entities1.intersection(entities2)

    @predicate
    def get(self, entity: Entity)-> Set[Entity]:
        """
        get entity and wrap it in a set (See Dialog-to-action Table 1 A15)

        """
        return {entity}

    @predicate
    def diff(self, entities1: Set[Entity], entities2: Set[Entity])-> Set[Entity]:
        """
        return instances included in entities1 but not included in entities2. Note that this is *NOT* the symmetric
        difference. E.g. set([1, 2]) - set([2, 3]) = set([1]) and *NOT* set([1, 2]) - set([2, 3]) = set([1, 3])

        """
        return entities1 - entities2

    @predicate
    def larger(self, entities: Set[Entity], predicate_: Predicate, num: Number)-> Set[Entity]:
        """
        subset of entities linking to more than num entities with predicate_
        """

        result = set()
        for entity in entities:
            try:
                linked_entities = self.kg_data[entity.id][predicate_.id]
                # if len(linked_entities) < num and linked_entities != 0:
                if len(linked_entities) > num:
                    # if n_links < num:
                    result.add(entity)
            except KeyError:
                continue

        return result

    @predicate
    def less(self, entities: Set[Entity], predicate_: Predicate, num: Number)-> Set[Entity]:
        """
        subset of entities linking to less than num entities with predicate_
        """

        result = set()

        cache_result: bool = self.search_modus

        if cache_result:
            try:
                count_cache = self.result_cache[self.function_cache.index(["less", entities, predicate_])]
                return set(ent for ent, count in count_cache.items() if count < num)
            except ValueError:
                count_cache = Counter()

        for entity in entities:
            try:
                linked_entities = self.kg_data[entity.id][predicate_.id]

                if len(linked_entities) < num:
                    # if n_links < num:
                    result.add(entity)

                if cache_result:
                    count_cache[entity] = len(linked_entities)
            except KeyError:
                continue

        if cache_result:
            self.function_cache.append(["less", entities, predicate_])
            self.result_cache.append(count_cache)

        return result

    @predicate
    def equal(self, entities: Set[Entity], predicate_: Predicate, num: Number)-> Set[Entity]:
        """
        subset of entities linking to exactly num entities with predicate_
        """
        result = set()
        for entity in entities:
            try:
                linked_entities = self.kg_data[entity.id][predicate_.id]
                # if len(linked_entities) < num and linked_entities != 0:
                if len(linked_entities) == num:
                    # if n_links < num:
                    result.add(entity)
            except KeyError:
                continue

        return result

    @predicate
    def most(self, entities: Set[Entity], predicate_: Predicate, num: Number)-> Set[Entity]:
        """
        subset of entities linking to at most num entities with predicate_
        """

        result = set()
        for entity in entities:
            try:
                linked_entities = self.kg_data[entity.id][predicate_.id]
                if len(linked_entities) <= num:
                    # if n_links < num:
                    result.add(entity)
            except KeyError:
                continue
        return result

    @predicate
    def most_with_type(self, entities: Set[Entity], predicate_: Predicate, num: Number, type_: Entity)-> Set[Entity]:
        """
        """

        result = set()
        type_predicate = self.get_predicate_from_question_id("P1", predicate_class=Predicate)

        for entity in entities:
            try:
                linked_entity_ids = self.kg_data[entity.id][predicate_.id]
                linked_entities = [self.get_entity_from_kg_id(ent_id) for ent_id in linked_entity_ids]
                linked_entities_with_type = self.has_relation_with(linked_entities, type_predicate, type_)

                if len(linked_entities_with_type) <= num:
                    # if n_links < num:
                    result.add(entity)
            except KeyError:
                continue
        return result

    @predicate
    def least(self, entities: Set[Entity], predicate_: Predicate, num: Number)-> Set[Entity]:
        """
        subset of entities linking to at least num entities with predicate_
        """

        result = set()
        for entity in entities:
            try:
                linked_entities = self.kg_data[entity.id][predicate_.id]
                if len(linked_entities) >= num:
                    # if n_links < num:
                    result.add(entity)
            except KeyError:
                continue
        return result
