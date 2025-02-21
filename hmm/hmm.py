import numpy as np

# Aided by Github Copilot, used reference from https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/

class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        

        #edge case empty
        if len(input_observation_states) == 0:
            return 0
        
        # Step 1. Initialize variables
        prob_table = np.zeros((len(input_observation_states), len(self.hidden_states)))
        for index, state in self.hidden_states_dict.items():
            prob_table[0, index] = self.prior_p[index] * self.emission_p[index][self.observation_states_dict[input_observation_states[0]]]
        
        # Step 2. Calculate probabilities
        for i in range(1, len(input_observation_states)):
            for j in range(len(self.hidden_states)):
                prob_table[i, j] = sum([prob_table[i-1, k] * self.transition_p[k, j] * self.emission_p[j, self.observation_states_dict[input_observation_states[i]]] for k in range(len(self.hidden_states))])


        # Step 3. Return final probability 
        forward_probability = sum(prob_table[-1])
        return forward_probability
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        

        #edge case empty
        if len(decode_observation_states) == 0:
            return []

        # Step 1. Initialize Variables
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        backpointer = np.zeros((len(decode_observation_states), len(self.hidden_states)), dtype=int)

        # Step 2. Calculate Probabilities
        for t in range(len(decode_observation_states)):
            for s in range(len(self.hidden_states)):
                if t == 0:
                    viterbi_table[t, s] = self.prior_p[s] * self.emission_p[s, self.observation_states_dict[decode_observation_states[t]]]
                else:
                    max_prob = max(viterbi_table[t-1, prev_s] * self.transition_p[prev_s, s] for prev_s in range(len(self.hidden_states)))
                    viterbi_table[t, s] = max_prob * self.emission_p[s, self.observation_states_dict[decode_observation_states[t]]]
                    backpointer[t, s] = np.argmax([viterbi_table[t-1, prev_s] * self.transition_p[prev_s, s] for prev_s in range(len(self.hidden_states))])

        # Step 3. Traceback and Find Best Path
        best_path_prob = max(viterbi_table[-1])
        best_path_pointer = np.argmax(viterbi_table[-1])
        best_path = [best_path_pointer]
        for t in range(len(decode_observation_states)-1, 0, -1):
            best_path.insert(0, backpointer[t, best_path[0]])

        # Step 4. Return Best Path
        return [self.hidden_states_dict[i] for i in best_path]
        