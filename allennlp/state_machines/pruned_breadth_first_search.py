import time
from collections import defaultdict
from allennlp.state_machines.states.grammar_based_search_state import GrammarBasedSearchState
from allennlp.semparse.domain_languages import START_SYMBOL


class PrunedBreadthFirstSearch:

    def __init__(self, language):
        language.set_search_mode()
        self.language = language

    def get_initial_state(self):
        language_valid_actions = defaultdict(list)
        for production_rule in self.language.all_possible_productions():
            lhs, rhs = production_rule.split(' -> ')
            language_valid_actions[lhs].append(rhs)
        return GrammarBasedSearchState(action_history=[], nonterminal_stack=[START_SYMBOL],
                                       valid_actions=language_valid_actions,
                                       is_nonterminal=self.language.is_nonterminal)

    def evaluates_to_result(self, actions, expected_result, verbose=False):
        start = time.time()
        #         print(language.action_sequence_to_logical_form(actions))
        query_result = self.language.execute_action_sequence(actions)
        evaluates_to_result = query_result == expected_result
        if verbose and time.time() - start > 0.5:
            duration = time.time() - start
            print("{} took {:.3f} seconds to execute".format(self.language.action_sequence_to_logical_form(actions),
                                                             round(duration, 2)))
        return evaluates_to_result

    def prune_by_depth(self, valid_actions, depth_left):
        return [v for v in valid_actions if v.count(',') + v.count(':') + 1 <= depth_left]

    def prune_first_action(self, valid_actions, depth, expected_result):
        if depth != 1:
            return valid_actions
        else:
            return [v for v in valid_actions if v == self.get_first_action_from_expected_result(expected_result)]

    def prune_successive_types(self, valid_actions, last_action):
        return [v for v in valid_actions if not (v in ["P1", "P-1"] and last_action in ["P1", "P-1"])]

    def prune_by_max_occurence(self, valid_actions, action_history, actions, max_):
        for action in actions:
            occurence = sum(action in a for a in action_history)
            valid_actions = [v for v in valid_actions if not (occurence >= max_ and v == action)]
        return valid_actions

    def prune_by_question_type(self, valid_actions, question, question_type, question_description):
        pruned_types = []

        if question_type in ["Quantitative Reasoning (All)", "Comparative Reasoning (All)"]:
            pruned_types += ["has_relation_with", "is_in"]
            if "More/Less" in question_description:
                pruned_types += ["equal", "equal_with_type", "least", "least_with_type", "most", "most_with_type",
                                 "min", "min_with_type", "max", "max_with_type"]
                if "less" in question or "lesser" in question:
                    pruned_types += ["more", "more_with_type"]
                elif "more" in question:
                    pruned_types += ["less", "less_with_type"]

            if "Atleast/ Atmost/ Approx. the same/Equal" in question_description:
                pruned_types += ["more", "more_with_type", "less", "less_with_type", "min", "min_with_type", "max",
                                 "max_with_type"]
                if "atmost" in question:
                    pruned_types += ["equal", "equal_with_type", "least", "least_with_type"]
                elif "atleast" in question:
                    pruned_types += ["equal", "equal_with_type", "most", "most_with_type"]

            if "Min/Max" in question_description:
                pruned_types += ["equal", "equal_with_type", "least", "least_with_type", "most", "most_with_type",
                                 "more", "more_with_type", "less", "less_with_type"]
                if "min" in question:
                    pruned_types += ["max", "max_with_type"]
                elif "max" in question:
                    pruned_types += ["min", "min_with_type"]
            if "Single entity type" in question_description:
                pruned_types += ["union", "intersection", "diff"]
        else:
            pruned_types += ["min", "min_with_type", "max", "max_with_type"]
            if "atleast" not in question:
                pruned_types += ["least", "least_with_type"]
            if "atmost" not in question:
                pruned_types += ["most", "most_with_type"]

            if "less" not in question and "lesser" not in question:
                pruned_types += ["less", "less_with_type"]
            if "more" not in question:
                pruned_types += ["more", "more_with_type"]
            if "exactly" not in question:
                pruned_types += ["equal", "equal_with_type"]

        return [v for v in valid_actions if v not in pruned_types]

    @staticmethod
    def prune_count_get(valid_actions, last_action, second_last_action):
        if second_last_action == "count" and last_action == "[<Entity:Set[Entity]>, Entity]":
            return [v for v in valid_actions if v != "get"]
        else:
            return valid_actions

    def prune_typed_questions(self, valid_actions, history, state, question_type_entities):
        # TODO: is it true that non_terminal_stack of 1 means that this is the last action?
        if len(history) > 2:
            contains_typed_function = history[2].split(" -> ")[1] in ["less_with_type", "more_with_type",
                                                                      "most_with_type", "least_with_type"]
            is_last_action = len(state._nonterminal_stack) == 1
            if contains_typed_function and is_last_action:
                return [v for v in valid_actions if v in question_type_entities]
        return valid_actions

    def get_last_action(self, action_history):
        if action_history:
            return action_history[-1].split(" -> ")[1]
        else:
            return None

    def get_second_last_action(self, action_history):
        if len(action_history) >= 2:
            return action_history[-2].split(" -> ")[1]
        else:
            return None

    def get_first_action_from_expected_result(self, expected_result):
        if isinstance(expected_result, set):
            return 'Set[Entity]'
        elif type(expected_result) == int:
            return 'Number'
        elif isinstance(expected_result, bool):
            return 'bool'
        else:
            raise ValueError()

    def print_search_result(self, action_sequences, depth, duration):
        print("\nfinished at depth {} in {} seconds".format(depth, duration))
        print([self.language.action_sequence_to_logical_form(a) for a in action_sequences])

    def search(self, question, question_type, question_description, question_type_entities, expected_result,
               verbose=False, max_depth=17, max_time=100, stop_after_n_found=1):

        states, correct_action_sequences = [self.get_initial_state()], []
        depth = 0
        finished = False
        start_search_time = time.time()
        while depth < max_depth:

            if time.time() - start_search_time > max_time or finished:
                break
            depth += 1

            next_states = []
            execution_time = 0.0
            for state in states:
                if state.is_finished():
                    start = time.time()
                    if self.evaluates_to_result(state.action_history, expected_result, verbose=verbose):
                        correct_action_sequences.append(state.action_history)
                        if len(correct_action_sequences) >= stop_after_n_found:
                            finished = True
                            break
                    execution_time += time.time() - start
                    continue

                actions_one = ["P-1", "union", "intersection", "diff", "equal", "equal_with_type", "least",
                               "least_with_type", "most", "most_with_type", "more", "more_with_type", "less",
                               "less_with_type", "min", "min_with_type", "max", "max_with_type"]

                history = state.action_history
                last_action = self.get_last_action(history)
                second_last_action = self.get_second_last_action(history)
                state_valid_actions = self.prune_by_depth(state.get_valid_actions(), max_depth - depth)
                state_valid_actions = self.prune_first_action(state_valid_actions, depth, expected_result)
                state_valid_actions = self.prune_successive_types(state_valid_actions, last_action)
                state_valid_actions = self.prune_by_max_occurence(state_valid_actions, history, actions_one, 1)
                state_valid_actions = self.prune_by_max_occurence(state_valid_actions, history, ["find"], 2)
                state_valid_actions = PrunedBreadthFirstSearch.prune_count_get(state_valid_actions, last_action, second_last_action)
                state_valid_actions = self.prune_by_question_type(state_valid_actions, question, question_type,
                                                                  question_description)
                state_valid_actions = self.prune_typed_questions(state_valid_actions, history, state,
                                                                 question_type_entities)

                if depth < max_depth:
                    for valid_action in state_valid_actions:
                        next_production_rule = state._nonterminal_stack[-1] + " -> " + valid_action
                        next_states.append(state.take_action(next_production_rule))
            states = next_states
            if execution_time > 0.1 and verbose:
                print(execution_time)

        if verbose:
            self.print_search_result(correct_action_sequences, depth, time.time() - start_search_time)

        return correct_action_sequences

