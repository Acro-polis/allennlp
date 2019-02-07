from allennlp.semparse import DomainLanguage
from allennlp.semparse.contexts import CSQAContext

from typing import Dict, List, NamedTuple, Set, Tuple
from numbers import Number


class CSQALanguage(DomainLanguage):
    # pylint: disable=too-many-public-methods,no-self-use
    """
    Implements the functions in the variable free language we use, that's inspired by the one in
    "Memory Augmented Policy Optimization for Program Synthesis with Generalization" by Liang et al.

    Because some of the functions are only allowed if some conditions hold on the table, we don't
    use the ``@predicate`` decorator for all of the language functions.  Instead, we add them to
    the language using ``add_predicate`` if, e.g., there is a column with dates in it.
    """
    def __init__(self, wikidata_context: CSQAContext) -> None:
        # TODO: do we need dates here too?
        super().__init__(start_types={Number, List[str]})

        self.kg_context = wikidata_context
        # Todo: Triple
        # self.kg_data = [Triple(triple) for triple in wikidata_context.kg_data]
        self.wikidata_graph = wikidata_context.get_knowledge_graph()

        # Adding entities and numbers seen in questions as constants.
        question_entities, question_numbers = wikidata_context.get_entities_from_question()

        self._question_entities = [entity for entity, _ in question_entities]
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
            self.terminal_productions[name] = "%s -> %s" % (types[0], name)
