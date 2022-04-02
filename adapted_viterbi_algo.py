#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
def viterbi():
    string = 'SMSL'
    nnodes = len(string)
    # Just general definitions for the gene problem
    alphabet = {'S': 0, 'M': 1, 'L': 2}
    table = {'H': [0.1, 0.4, 0.5],
             'C': [0.7, 0.2, 0.1]
             }
    translation = {'H': 'H', 'C': 'C'}
    # Create the list of possible states, which is just a state per section
    x_alphabet = ['H', 'C']
    x_len = len(x_alphabet)
    # Create the transition matrix between states
    matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
    # Initialize all messages to empty
    fwd = np.ones((x_len, nnodes))
    bwd = np.ones((x_len, nnodes))
    fwd[0, 0] = 0.6
    fwd[1, 0] = 0.4
    # Compute the message for each node
    for i in range(nnodes - 1):
        # Compute the forward message
        fwd_index = i + 1
        # For each possible value of i+1
        for x_idx2, x2 in enumerate(x_alphabet):
            s = 0
            # For each possible value of i
            for x_idx1, x1 in enumerate(x_alphabet):
                p = node_potential(x1, string[i], table, alphabet)  # Compute the potential
                p *= matrix[x_idx1, x_idx2]  # Multiply by the edge transition probability
                p *= fwd[x_idx1, fwd_index - 1]  # Multiply by the previous message
                s = max(s, p)  # Only select the maximum probability message
            fwd[x_idx2, fwd_index] = s  # Update the current message
        # Compute the backward message, note that we start at the end and go backwards
        bwd_index = nnodes - i - 2
        # For each possible value of i-1
        for x_idx2, x2 in enumerate(x_alphabet):
            s = 0
            # For each possible value of i
            for x_idx1, x1 in enumerate(x_alphabet):
                p = node_potential(x1, string[bwd_index + 1], table, alphabet)  # Compute the potential
                p *= matrix[x_idx2, x_idx1]  # Multiply by the edge transition probability
                p *= bwd[x_idx1, bwd_index + 1]  # Multiply by the previous message
                s = max(s, p)  # Only select the maximum probability message
            bwd[x_idx2, bwd_index] = s  # Update the current message
    # At this point we have the forward and backward messages computed, we only need to infere the most probable
    # region for each gene
    mpc = [x_alphabet[0] for _ in range(nnodes)]  # Initialize the most probable to empty
    probs = np.zeros((nnodes, x_len))
    for i in range(nnodes):
        # For each node, compute the probability of each character in the alphabet and keep the most probable
        for j, x in enumerate(x_alphabet):
            value = node_potential(x, string[i], table, alphabet)  # Compute the potential
            value *= fwd[j, i]  # Multiply by the forward message
            value *= bwd[j, i]  # Multiply by the backward message
            probs[i, j] = value  # Update the probability matrix
    # For each node, search the most probable value
    for i in range(nnodes):
        j = int(np.argmax(probs[i, :]))
        mpc[i] = translation[x_alphabet[j]]
    # Now just create the final string based on this value
    new_string = ''
    count = 0
    for s in mpc:
        new_string += s
        count += 1
        if count % 10 == 0:
            new_string += '\n'
    if new_string[-1] == '\n':  # Erase the last character if empty
        new_string = new_string[:-1]
    return new_string
print(viterbi())

