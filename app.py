import streamlit as st
import numpy as np
import gym
from matplotlib import pyplot as plt
import time
import os

st.title("Deep Learning")

test = st.button("Test")

if test:
    execfile("game.py")
