from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from hashlib import sha256
import os
from tkinter import ttk
import time
import pandas as pd
from keras.models import load_model
from Agent import *
import math
import numpy as np
import random
from collections import deque
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

main = Tk()
main.title("Improving TCP Congestion Control with Machine Intelligence")
main.geometry("1300x1200")

global filename
lp_arr = []
reno_arr = []
rewards = []
penalty = []
global dataset, records, loss_value, throughput_value

def uploadDataset():
    global dataset, records, filename
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename)
    records = dataset.shape[0]
    text.insert(END,str(dataset.head()))

def preprocessDataset():
    global dataset, records
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    text.insert(END,"Dataset Processing Completed\n\n")
    text.insert(END,"Total records found in dataset : "+str(records))
 
def formatLoss(qty):
    return str(qty)

def getNetworkLoss():
    global dataset
    dataset = dataset.values
    dataset = dataset[0:150]
    consumption = dataset[:,dataset.shape[1]-1]
    temp = []
    for i in range(len(consumption)):
        temp.append(consumption[i])
    consumption = temp    
    print(consumption)
    return consumption 

def calculateSigmoid(gamma_value):
    if gamma_value < 0:
        return 1 - 1/(1 + math.exp(gamma_value))
    else:
        return 1/(1 + math.exp(-gamma_value))
    

def getAgentState(values, index, total):
    data = index - total + 1
    block = values[data:index + 1] if data >= 0 else -data * [values[0]] + values[0:index + 1]
    result = []
    for i in range(total - 2):
        result.append(calculateSigmoid(block[i + 1] - block[i]))
    return np.array([result])

def trainRL():
    global lp_arr, reno_arr, rewards, penalty, loss_value, throughput_value
    loss_value = []
    throughput_value = []
    lp_arr.clear()
    reno_arr.clear()
    text.delete('1.0', END)
    windowSize = 10
    episodeCount = 1000
    windowSize = int(windowSize)
    episodeCount = int(episodeCount)
    count = 0
    agent = Agent(windowSize) #creating agent object
    loss = getNetworkLoss() #get network loss
    length = len(loss) - 1
    batchSize = 32
    modelName = 'models/model_ep500'
    model = load_model(modelName) #load the model
    windowSize = model.layers[0].input.shape.as_list()[1]
    state = getAgentState(loss, 0, windowSize + 1)#based on environment agent will calculate state
    cuttent_Quantity = 0
    agent.network = []
    rewards.clear()
    penalty.clear()
    penalty_value = 0
    for t in range(length):
        action = agent.act(state) #now agent will act and take action
        nextState = getAgentState(loss, t + 1, windowSize + 1)
        rewardValue = 0
        if action == 1: # go for buy and decision is accuarte and add reward values
            agent.network.append(loss[t])
            lp_arr.append(loss[t])
        
        elif action == 2 and len(agent.network) > 0: #check loss
            old_loss = agent.network.pop(0)
            rewardValue = max(loss[t] - old_loss, 0)
            current_loss = old_loss - loss[t] 
            if  old_loss > loss[t]:
                reno_arr.append(old_loss - loss[t])
            else: 
                reno_arr.append(loss[t] - old_loss)
        done = True if t == length - 1 else False
        if rewardValue != 0:
            rewards.append(rewardValue)
        else: #action is not accurate and add penalty
            if len(penalty) < 30:
                penalty_value = penalty_value + 0.2
                penalty.append(penalty_value)
        agent.memory.append((state, action, rewardValue, nextState, done))
        state = nextState
        if done:
            current_loss = current_loss / 100
            text.insert(END,"\n--------------------------------\n")
            text.insert(END,"RL-TCP Current Loss       : " + formatLoss(current_loss)+"\n")
            text.insert(END,"RL-TCP Current Throughput : " + formatLoss(1 - current_loss)+"\n\n")
            loss_value.append(current_loss)
            throughput_value.append((1 - current_loss))
            print("--------------------------------")
    lp_arr = np.asarray(lp_arr)       
    reno_arr = np.asarray(reno_arr)
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Simulation Time')
    plt.ylabel('CWND Num Packets')
    plt.plot(lp_arr, 'ro-', color = 'green')
    plt.plot(reno_arr, 'ro-', color = 'blue')
    plt.legend(['LP-TCP', 'New-Reno'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('CWND Num Packets Graph')
    plt.show()    

def trainLP():
    global lp_arr, reno_arr, rewards, penalty, loss_value, throughput_value, filename
    dataset = pd.read_csv(filename)#read dataset
    dataset.fillna(0, inplace = True)#remove missing values
    delay = dataset['global_delay'].ravel()
    dataset.drop(['jitter', 'global_delay' ,'global_losses'], axis = 1,inplace=True)
    X = dataset.values[:,0:dataset.shape[1]]
    X_train, X_test, y_train, y_test = train_test_split(X, delay, test_size = 0.5, random_state = 0)
    lp = RandomForestRegressor(n_estimators = 20)
    lp.fit(X_train, y_train)
    predict = lp.predict(X_test)
    loss = mean_squared_error(y_test, predict)
    text.insert(END,"LP-TCP Current Loss       : " + formatLoss(loss)+"\n")
    text.insert(END,"LP-TCP Current Throughput : " + formatLoss(1 - loss)+"\n")
    loss_value.append(loss)
    throughput_value.append((1 - loss))
    predict = predict[0:len(reno_arr)]

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Simulation Time')
    plt.ylabel('CWND Num Packets')
    plt.plot(predict, 'ro-', color = 'green')
    plt.plot(reno_arr, 'ro-', color = 'blue')
    plt.legend(['LP-TCP', 'New-Reno'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('CWND Num Packets Graph')
    plt.show()
    
def graph():
    df = pd.DataFrame([['Loss','LP-TCP',loss_value[1]],['Loss','RL-TCP',loss_value[0]],
                       ['Throughput','LP-TCP',throughput_value[1]],['Throughput','RL-TCP',throughput_value[0]],
                        
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()
    

font = ('times', 15, 'bold')
title = Label(main, text='Improving TCP Congestion Control with Machine Intelligence')
title.config(bg='bisque', fg='purple1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')

uploadButton = Button(main, text="Upload Network Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=350,y=100)
processButton.config(font=font1)

lpButton = Button(main, text="Run LP-TCP Algorithm", command=trainLP)
lpButton.place(x=350,y=150)
lpButton.config(font=font1)

rlButton = Button(main, text="Run RL-TCP Algorithm", command=trainRL)
rlButton.place(x=50,y=150)
rlButton.config(font=font1)

graphButton = Button(main, text="Loss & Throughput Graph", command=graph)
graphButton.place(x=650,y=150)
graphButton.config(font=font1)

font1 = ('times', 13, 'bold')
text=Text(main,height=25,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

main.config(bg='cornflower blue')
main.mainloop()
