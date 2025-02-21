import pytest
from hmm import HiddenMarkovModel
import numpy as np

#Assisted by Github Copilot

def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    mini_hmm_model = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], mini_hmm['prior_p'], mini_hmm['transition_p'], mini_hmm['emission_p'])
   
    mini_input_observation_states = mini_input['observation_state_sequence']
    mini_input_hidden_states = mini_input['best_hidden_state_sequence']

    forward_probability = mini_hmm_model.forward(mini_input_observation_states)
    assert np.isclose(forward_probability, 0.03506, atol=1e-5)

    assert(list(mini_input_hidden_states) == mini_hmm_model.viterbi(mini_input_observation_states))

    #Edge case 1: When the observation sequence is empty
    empty_observation_states = np.array([])
    assert mini_hmm_model.forward(empty_observation_states) == 0
    assert mini_hmm_model.viterbi(empty_observation_states) == []

    #Edge case 2: When the observation is not one that is in the model
    wrong_observation_states = np.array(['cloudy'])
    with pytest.raises(KeyError):
        mini_hmm_model.forward(wrong_observation_states)
    with pytest.raises(KeyError):
        mini_hmm_model.viterbi(wrong_observation_states)

def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    big_hmm_model = HiddenMarkovModel(full_hmm['observation_states'], full_hmm['hidden_states'], full_hmm['prior_p'], full_hmm['transition_p'], full_hmm['emission_p'])

    full_input_observation_states = full_input['observation_state_sequence']
    full_input_hidden_states = full_input['best_hidden_state_sequence']

    assert(list(full_input_hidden_states) == big_hmm_model.viterbi(full_input_observation_states))













