#! /usr/bin/env python

# pylint: disable=invalid-name,wrong-import-position

import sys
import os
import time
import argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.common import Params
from allennlp.data.dataset_readers.semantic_parsing.csqa.csqa import CSQADatasetReader
from allennlp.state_machines.states.grammar_statelet import GrammarStatelet
from allennlp.state_machines.states.grammar_based_search_state import GrammarBasedSearchState
from allennlp.semparse.domain_languages.csqa_language import CSQALanguage
from allennlp.semparse.domain_languages.csqa_language import Entity, Predicate
from allennlp.semparse.domain_languages import START_SYMBOL

from pathlib import Path


def search(csqa_directory: str,
           wikidata_directory: str,
           max_path_length: int,
           max_num_logical_forms: int) -> None:

    # create output dir
    output_directory = f'{csqa_directory}/logical_forms'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    params = {'lazy': False,
              #'kg_path':  f'{AllenNlpTestCase.FIXTURES_ROOT}/data/csqa/sample_kg.p',
              'kg_path':  f'{wikidata_directory}/wikidata_short_1_2.p',
              'entity_id2string_path':  f'{AllenNlpTestCase.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json',
              'predicate_id2string_path': f'{AllenNlpTestCase.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
             }

    reader = CSQADatasetReader.from_params(Params(params))
    qa_path = f'{AllenNlpTestCase.FIXTURES_ROOT}/data/csqa/sample_qa.json'
    dataset = reader.read(qa_path)
    print_instance_info = True
    logical_forms_found = 0

    for instance in dataset[2:5]:
        instance_start_time = time.time()

        question: str = instance['question'].tokens
        language: CSQALanguage = instance['world'].metadata
        expected_result = instance['expected_result'].metadata

        # initialize states
        valid_actions = defaultdict(list)
        for production_rule in language.all_possible_productions():
            lhs, rhs = production_rule.split(' -> ')
            valid_actions[lhs].append(rhs)
        initial_grammar_statelet = GrammarStatelet(nonterminal_stack=[START_SYMBOL], valid_actions=valid_actions,
                                                   is_nonterminal=language.is_nonterminal)
        initial_state = GrammarBasedSearchState(action_history=[],
                                                grammar_state=initial_grammar_statelet)

        states = [initial_state]
        finished_states, correct_states = [], []
        depth = 0
        evaluated_states = 0

        while depth < 11:
            depth += 1
            next_states = []
            for state in states:
                evaluated_states += 1
                start_time = time.time()
                if state.is_finished():
                    finished_states.append(state)
                    # if set(language.execute_action_sequence(state.action_history)) == set(result_entities):
                    query_result = language.execute_action_sequence(state.action_history)
                    query_result = set(query_result) if isinstance(query_result, list) else query_result
                    if query_result == expected_result:
                        correct_states.append(state)
                    continue
                if (time.time()-start_time) > 1.0:
                    print(time.time()-start_time)
                    print(state.action_history)

                for valid_action in state.get_valid_actions():
                    production_rule = state.grammar_state._nonterminal_stack[-1] + " -> " + valid_action
                    next_states.append(state.take_action(production_rule))

            states = next_states

        if len(correct_states) > 0:
            logical_forms_found +=1

        # print instance stuff
        if print_instance_info:
            print("\n" + "="*100)
            if len(correct_states) == 0:
                print("COULD NOT FIND ANSWER: {} (SHOWING SAMPLE!)".format(
                    expected_result if not isinstance(expected_result, set) else list(expected_result)[0:3]))
            else:
                correct_sequences = [s.action_history for s in correct_states]
                correct_predictions = [language.execute_action_sequence(seq) for seq in correct_sequences]
                print("found correct answer: {}".format(correct_predictions[0]))
                print("{} sequence(s) found {}".format(len(correct_sequences), correct_sequences))

            print("{} evaluated states in {:.2f} seconds ({} finished)".format(evaluated_states, time.time() - instance_start_time,
                                                                               len(finished_states)))
            print("question: {}".format(question))
            print("="*100)

    print(logical_forms_found)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csqa_directory", type=str, help="Path to the csqa data root")
    parser.add_argument("--wikidata_directory", type=str, help="Path to the wikidata root")
    parser.add_argument("--max_path_length", type=int, dest="max_path_length", default=10,
                        help="Max length to which we will search exhaustively")
    parser.add_argument("--max_num_logical-forms", type=int, dest="max_num_logical_forms",
                        default=100, help="Maximum number of logical forms returned")


    args = parser.parse_args()
    search(args.csqa_directory, args.wikidata_directory, args.max_path_length, args.max_num_logical_forms)
