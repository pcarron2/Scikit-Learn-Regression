import numpy as np
import matplotlib.pyplot as plt

#Create values and labels for bar chart
values =np.random.rand(3)
inds   =np.arange(3)
labels = ["A","B","C"]

#Plot a bar chart
plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.bar(inds, values, align='center') #This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("Error") #Y-axis label
plt.xlabel("Method") #X-axis label
plt.title("Error vs Method") #Plot title
plt.xlim(-0.5,2.5) #set x axis range
plt.ylim(0,1) #Set yaxis range

#Set the bar labels
plt.gca().set_xticks(inds) #label locations
plt.gca().set_xticklabels(labels) #label values

#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("../Figures/example_bar_chart.pdf")


#Create values and labels for line graphs
values =np.random.rand(2,5)
inds   =np.arange(5)
labels =["Method A","Method B"]

#Plot a line graph
plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.plot(inds,values[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
plt.plot(inds,values[1,:],'sb-', linewidth=3) #Plot the first series in blue with square marker

#This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("Error") #Y-axis label
plt.xlabel("Value") #X-axis label
plt.title("Error vs Value") #Plot title
plt.xlim(-0.1,4.1) #set x axis range
plt.ylim(0,1) #Set yaxis range
plt.legend(labels,loc="best")

#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("../Figures/example_line_plot.pdf")

#Displays the plots.
#You must close the plot window for the code following each show()
#to continue to run
plt.show()
plt.clf()  # clear figure






#Displays the charts.
#You must close the plot window for the code following each show()
#to continue to run
plt.show()
plt.clf()  # clear figure
