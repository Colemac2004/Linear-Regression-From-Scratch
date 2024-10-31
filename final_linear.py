import csv
import math
import random



#read csv
def read_csv(filename):
    with open(filename,mode='r',newline='') as file:
        csv_file=csv.reader(file)
        return list(csv_file)

#normalize the data
def normalize(csv_file):
    uniques={'sex':{},'smoker':{},'region':{}}
    #iterate through row
    for row in csv_file:
        for key,index in zip(['sex','smoker','region'],[1,4,5]):
            value=row[index]
            if value not in uniques[key]:
                 uniques[key][value] = len(uniques[key])
            row[index]=uniques[key][value]
        for c in [0,2,3,6]:
            row[c]=float(row[c])
    return csv_file

#get y
def get_y(csv_file):
    y_list=[]
    for row in csv_file:
        y_list.append(row[6])
        row.pop(6)
    return y_list,csv_file

#get mins and maxs
def min_max(csv_file):
    maxs={f'max_{i}':-math.inf for i in range(1,7)}
    mins={f'min_{i}':math.inf for i in range(1,7)}
    for row in csv_file:
        for i in range(0,6):
            if float(row[i]) > maxs[f'max_{i+1}']:
                maxs[f'max_{i+1}']=float(row[i])
            if float(row[i]) < mins[f'min_{i+1}']:
                mins[f'min_{i+1}']=float(row[i])
    return mins,maxs


#def scale data
def scale_data(csv_file,mins,maxs):
    for i in range(0,len(csv_file)):
        for x in range(0,6):
            csv_file[i][x]=round(((float(csv_file[i][x])-mins[f'min_{x+1}'])/(maxs[f'max_{x+1}']-mins[f'min_{x+1}'])),5)
    return csv_file



#read csv and pop row 1
csv_file=read_csv('insurance.csv')
csv_file.pop(0)

#normalize
normalized_data=normalize(csv_file)

#get y
y_list,data=get_y(normalized_data)

#mins and maxs
mins,maxs=min_max(data)

#scale data
scaled_data=scale_data(data,mins,maxs)

#now we fit model
#initalize weights and bias
weights={f'w{i}':0 for i in range (1,7)}
bias=1
#declare part of data
part_data=scaled_data[:round((.8)*len(scaled_data))]
for i in range(500):
    #for regular gradient decent
    learning_rate=0.1
    total_loss=0
    bias_adj=0
    for c in range(0,len(part_data)):
        #calculate loss
        row_val=0
        loss=0
        for x in range(0,6):
            row_val+=(part_data[c][x]*weights[f'w{x+1}'])
        loss+=((y_list[c]-(row_val+bias))**2)
        total_loss+=loss
        #define weight adjustments
        weight_adjustment={f'w_adj{i}':0 for i in range(1,7)}
        #now for partial derivatives
        for b in range(0,6):
            weight_adjustment[f'w_adj{b+1}']+=part_data[c][b]*loss
        #now for bias    
        bias_adj+=loss
    #now average the partials out
    for i in range(1,7):
        weight_adjustment[f'w_adj{i}']=weight_adjustment[f'w_adj{i}']*(-2/len(part_data))
        weights[f'w{i}']=weights[f'w{i}']-(learning_rate*weight_adjustment[f'w_adj{i}'])
    bias_adj=bias_adj*(-2/len(part_data))
    bias=bias-(learning_rate*bias_adj)
    #adjust weights
    #now we average loss
    mse=total_loss/len(part_data)
    print(mse)

