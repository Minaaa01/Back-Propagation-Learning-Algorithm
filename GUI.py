from tkinter import *
from tkinter import ttk, Checkbutton, Entry

Activation_Function_list = [
    "Sigmoid",
    "Hyperbolic Tangent sigmoid",
]
Userinfo = []
Number_Of_Neurons_list = []


def Get_Number_Of_Hidden_Layers():
    Userinfo.append(Hidden_Layers.get())


def Get_Number_Of_Neurons():
    Number_Of_Neurons_list.append(NumberOf_Neurons.get().split(','))


def Get_ETA_Value():
    Userinfo.append(ETA_Value.get())


def Get_Number_Of_Epochs_Value():
    Userinfo.append(Number_Of_Epochs_Value.get())


def Get_Bias_Value():
    Userinfo.append(Bias_Value.get())


def MyFirstCombo_(event):
    Userinfo.append(MyFirstCombo.current())
    print("The parameters that used for these results is : ")
    print(f'number of hidden layers: {Userinfo[0]}')
    print(f'number of neurons in each hidden layer: {Number_Of_Neurons_list}')
    print(f'learning rate (eta): {Userinfo[1]}')
    print(f'number of epochs (m): {Userinfo[2]}')
    print(f'Bias: {Userinfo[3]}')
    print(f'Activation Functions: {MyFirstCombo.get()}')
    print(50 * "*")


master = Tk(className=" MLP ")
master.geometry('500x500')

HL = StringVar()
i = Label(master, textvariable=HL)
HL.set("Enter number of hidden layers")
i.place(x=20, y=30)

Hidden_Layers = Entry(master)
Hidden_Layers.place(x=190, y=30)
Hidden_Layers.focus_set()
Hidden_Layers_ = Button(master, text="Enter", width=8, borderwidth=4, command=Get_Number_Of_Hidden_Layers)
Hidden_Layers_.place(x=320, y=25)

NONeurons = StringVar()
i2 = Label(master, textvariable=NONeurons)
NONeurons.set("Enter number of neurons")
i2.place(x=20, y=80)

NumberOf_Neurons = Entry(master)
NumberOf_Neurons.place(x=190, y=80)
NumberOf_Neurons.focus_set()

Number_Of_Neurons_ = Button(master, text="Enter", width=8, borderwidth=4, command=Get_Number_Of_Neurons)
Number_Of_Neurons_.place(x=320, y=75)

ETA = StringVar()
i5 = Label(master, textvariable=ETA)
ETA.set("Enter learning rate (eta)")
i5.place(x=20, y=130)

ETA_Value = Entry(master)
ETA_Value.place(x=190, y=130)
ETA_Value.focus_set()
ETA_Button = Button(master, text="Enter", width=8, borderwidth=4, command=Get_ETA_Value)
ETA_Button.place(x=320, y=125)

Number_Of_Epochs = StringVar()
i6 = Label(master, textvariable=Number_Of_Epochs)
Number_Of_Epochs.set("Enter number of epochs (m)")
i6.place(x=20, y=185)

Number_Of_Epochs_Value = Entry(master)
Number_Of_Epochs_Value.place(x=190, y=185)
Number_Of_Epochs_Value.focus_set()
Number_Of_Epochs_Button = Button(master, text="Enter", width=8, borderwidth=4, command=Get_Number_Of_Epochs_Value)
Number_Of_Epochs_Button.place(x=320, y=180)

Bias_Value = IntVar()
Check_Box = Checkbutton(master, text="Add bias or not", variable=Bias_Value, offvalue=0)
Check_Box.place(x=20, y=220)
Bias_Button = Button(master, text="Add", width=10, borderwidth=4, command=Get_Bias_Value)
Bias_Button.place(x=22, y=245)

Activation_Functions = StringVar()
i5 = Label(master, textvariable=Activation_Functions)
Activation_Functions.set("Choose Your Activation Functions")
i5.place(x=20, y=285)

MyFirstCombo = ttk.Combobox(master, state="readonly", width=25, values=Activation_Function_list)
MyFirstCombo.bind("<<ComboboxSelected>>", MyFirstCombo_)
MyFirstCombo.place(x=22, y=315)

master.mainloop()
