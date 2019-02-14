from allennlp.semparse.contexts import CSQAContext
from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, ExecutionError,
                                                                predicate)
import logging

from typing import Dict, List, NamedTuple, Set, Tuple
from numbers import Number

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Predicate(NamedTuple):
    name: str


class CSQALanguage(DomainLanguage):
    # pylint: disable=too-many-public-methods,no-self-use
    """
    Implements the functions in the variable free language in "Dialog-to-Action: Conversational Question
    Answering Over a Large-Scale Knowledge Base" by Daya Guo, Duyu Tang, Nan Duan, Ming Zhou, and Jian Yin
    """
    def __init__(self, csqa_context: CSQAContext) -> None:
        # TODO: do we need dates here too?
        # TODO: check name and value passed to add_constant
        super().__init__(start_types={Number, List[str]})
        self.kg_context = csqa_context
        self.kg_data = csqa_context.kg_data

        for id, predicate in csqa_context.predicate_id2string.items():
            self.add_constant(id, id)
        question_entities, question_numbers = csqa_context.get_entities_from_question()

        self._question_entities = question_entities
        self._question_numbers = [number for number, _ in question_numbers]

        for entity in self._question_entities:
            self.add_constant(entity, entity)

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


    def get_agenda(self):
        # TODO: this needs to be implemented when we are searching for logical forms
        raise NotImplementedError("")

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
    def all_entities(self) -> List[str]:
        """
        Get all entities in KG
        """
        return list(self.kg_data.keys())

    @predicate
    def find(self, entities: List[str], predicate_: str) -> List[str]:
        """
        find function takes a list of entities E and and a predicate p and loops through
        e in E and returns the set of entities with a p edge to e
        """

        """Get the property of a list of entities."""
        result = set()
        for ent in entities:
            try:
                result = result.union(self.kg_data[ent][predicate_])
            except KeyError:
                continue
        return list(result)

    @predicate
    def count(self, entities: List[str]) -> Number:
        return len(entities)  # type: ignore

    @predicate
    def is_in(self, entity: str, entities: List[str]) -> Number:
        """
        return whether the first entity is in the set of entities

        """
        return entity in entities

    @predicate
    def union(self, entities1: List[str], entities2: List[str]) -> List[str]:
        """
        return union of two sets of entities

        """
        return list(set(entities1).union(entities2))

    @predicate
    def intersection(self, entities1: List[str], entities2: List[str]) -> List[str]:
        """
        return intersection of two sets of entities

        """
        return list(set(entities1).intersection(entities2))

    @predicate
    def get(self, entity: str)-> List[str]:
        """
        get entity and wrap it in a set (See Dialog-to-action Table 1 A15)

        """
        return [entity]

    @predicate
    def diff(self, entities1: List[str], entities2: List[str])-> List[str]:
        """
        return instances included in entities1 but not included in entities2. Note that this is *NOT* the symmetric
        difference. E.g. set([1, 2]) - set([2, 3]) = set([1]) and *NOT* set([1, 2]) - set([2, 3]) = set([1, 3])

        """
        return list(set(entities1) - set(entities2))

    @predicate
    def larger(self, entities: List[str], predicate_: str, num: Number)-> List[str]:
        """
        subset of entities linking to more than num entities with predicate_
        """

        result = set()
        for entity in entities:
            if len(self.find([entity], predicate_)) > num:
                result = result.union([entity])

        return list(result)

    @predicate
    def less(self, entities: List[str], predicate_: str, num: Number)-> List[str]:
        """
        subset of entities linking to less than num entities with predicate_
        """
        # TODO (koen): do we want to include entities that have 0 relations?
        result = set()
        for entity in entities:
            if len(self.find([entity], predicate_)) < num:
                result = result.union([entity])

        return list(result)

    @predicate
    def equal(self, entities: List[str], predicate_: str, num: Number)-> List[str]:
        """
        subset of entities linking to exactly num entities with predicate_
        """

        result = set()
        for entity in entities:
            if len(self.find([entity], predicate_)) == num:
                result = result.union([entity])

        return list(result)

    @predicate
    def most(self, entities: List[str], predicate_: str, num: Number)-> List[str]:
        """
        subset of entities linking to at most num entities with predicate_
        """

        result = set()
        for entity in entities:
            if len(self.find([entity], predicate_)) <= num:
                result = result.union([entity])

        return list(result)

    @predicate
    def least(self, entities: List[str], predicate_: str, num: Number)-> List[str]:
        """
        subset of entities linking to at least num entities with predicate_
        """

        result = set()
        for entity in entities:
            if len(self.find([entity], predicate_)) >= num:
                result = result.union([entity])

        return list(result)

