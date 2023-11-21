import numpy as np
import math
import random
# Define your objective function here
def objective_function(x,question_no):
    match question_no:
        case 1:
            return -((math.pow((2*x-5),4)) - math.pow((x**2 - 1),3))
        case 2:
            return -(8+x**3 -2*x -2 *math.exp(x))
        case 3:
            return -((4*x)*math.sin(x))
        case 4:
            return 2*(math.pow(x-3,2)) + math.exp((0.5*(x*x)))
        case 5:
            return x*x - 10* math.exp(0.1*x)
        case 6:
            return -(20*math.sin(x) - 15*(x**2))
        case 7:
            return (x*x + (54/x))



#Interval Halving method
def interval_halving_method(a, b, question_no,epsilon = 1e-4):
    
    print("*********************************************")
    print("Interval Halving Method")
    # epsilon = float(input("Enter the epsilon value(by default 10^-3): "))
    #value of xm
    x_m = (a+b)/2
    #length
    l = b-a
    #f(xm) function evaluation
    f_x_m = objective_function(x_m,question_no)
    #counter for function evaluations
    f_eval = 1
    #counter for num_iterations
    iter = 1
    #create a file according to the question/function number
    out = open(r"Interval_Halving_Method_"+str(question_no)+".txt", "w")  # Output file
    out.write("*********************************************************************************")
    out.write("\n")
    
    out.write("#It\t\t'a'\t\t'x1'\t\t'x2'\t\t'b'\t\t'xm'\t\t'f(x1)'\t\t'f(x2)'\t\t'f(xm)'")
    out.write("\n")
    out.write(str(iter)+"\t\t"+str(round(a,4))+"\t\t"+"nil"+"\t\t"+"nil"+"\t\t"+str(round(b,4))+"\t\t"+str(round(x_m,4))+"\t\t"+"nil"+"\t\t"+"nil"+"\t\t"+str(round(f_x_m,4)))
    #if |l|<epsilon   we break
    while(abs(l)>= epsilon):
        out.write("\n")
        #step 2
        x_1 = a+l/4
        x_2 = b-l/4
        f_x_1 = objective_function(x_1,question_no)
        f_x_2 = objective_function(x_2,question_no)
        f_eval+=2
        #2 more evaluations for each iterations
        #step 3
        if(f_x_1 < f_x_m):
            b= x_m
            x_m = x_1
            f_x_m = f_x_1
        elif(f_x_2 < f_x_m):
            a=x_m
            x_m = x_2
            f_x_m  = f_x_2
        else:
            a=x_1
            b=x_2
        #step 4 new length
        l=b-a
        iter+=1
        #writting to the log file
        out.write(str(iter)+"\t\t"+str(round(a,4))+"\t\t"+str(round(x_1,4))+"\t\t"+str(round(x_2,4))+"\t\t"+str(round(b,4))+"\t\t"+str(round(x_m,4))+"\t\t"+str(round(f_x_1,4))+"\t\t"+str(round(f_x_2,4))+"\t\t"+str(round(f_x_m,4)))

    
    out.write("\n")
    out.write("\n")
    # Overall Result
    out.write("The minimum point lies between  "+ str(round(a,4)) +" and " + str(round(b,4)))    # Store in the file
    out.write("\n")
    out.write("Total number of function evaluations: " + str(round(f_eval,4)))
    out.write("\n")
    
    out.write("*********************************************************************************")
    out.write("\n")
    
    out.write("*********************************************************************************")
    out.close()
    print("*********************************************")
    return a,b





def bounding_phase_method(a,b, question_no,delta = 0.02):
    k = 0
    #step 1 set k = 0 , choose initial guess
    print("*********************************************")
    print("Bounding Phase Method")
    x_0 = (a+b)/2
    while(1):
        # x_0 = float(input("Enter initial guess: "))
        x_0 = random.randrange(int(a),int(b)+1)
        f_x_0 = objective_function(x_0,question_no)
        f_x_0_del_minus = objective_function((x_0-delta),question_no)
        f_x_0_del_plus = objective_function((x_0+delta),question_no)
        if(f_x_0_del_minus<=f_x_0<=f_x_0_del_plus):
            delta = -delta
            break
        elif(f_x_0_del_minus>=f_x_0>=f_x_0_del_plus):
            break
        else:
            print("Choose some other initial guess: ")



    x_k =x_k_minus_1 = x_0
    x_k_plus_1 = x_k + ((2**k)*(delta))
    f_x_k_plus_1 = objective_function(x_k_plus_1,question_no)
    f_x_k = objective_function(x_k,question_no)


    f_eval = 2
    iter = 1
    out = open(r"Bounding_Phase_Method_"+str(question_no)+".txt", "w")  # Output file
    out.write("*********************************************************************************")
    out.write("\n")
    
    out.write("#It\t\tx(k-1)\t\tx(k)\t\tx(k+1)\t\tf(x(k))\t\tf(x(k+1))")
    out.write("\n")
    out.write(str(iter)+"\t\t"+str(round(x_k_minus_1,4))+"\t\t"+str(round(x_k,4))+"\t\t"+str(round(x_k_plus_1,4))+"\t\t"+str(round(f_x_k,4))+"\t\t"+str(round(f_x_k_plus_1,4)))
    out.write("\n")
    

    while(f_x_k_plus_1<f_x_k):
        k+=1
        iter +=1
        x_k_minus_1 = x_k
        x_k = x_k_plus_1
        x_k_plus_1 = x_k + ((2**k)*(delta))
        f_x_k  = f_x_k_plus_1
        f_x_k_plus_1 = objective_function(x_k_plus_1,question_no)
        out.write(str(iter)+"\t\t"+str(round(x_k_minus_1,4))+"\t\t"+str(round(x_k,4))+"\t\t"+str(round(x_k_plus_1,4))+"\t\t"+str(round(f_x_k,4))+"\t\t"+str(round(f_x_k_plus_1,4)))
        out.write("\n")
        f_eval+=1

    x = min(x_k_minus_1,x_k_plus_1)
    y = max(x_k_minus_1,x_k_plus_1)


    out.write("\n")
    out.write("The minimum point lies between  "+ str(round(x_k_minus_1,4)) +" and " + str(round(x_k_plus_1,4)))    # Store in the file
    out.write("\n")
    out.write("Total number of function evaluations: " + str(round(f_eval,4)))
    out.write("\n")
    out.write("\n")
    
    out.write("****************************************************************************************")

    out.close()
    print("*********************************************")
    return round(x,4),round(y,4)

    



# Main program
def main_program():
    # Ask for user input for the question/fuunction number
    question_no = int(input("Enter the Question Number/ Function: "))
    print("\n**********************************")
        #input a and b 
    a = float(input("Enter the initial lower bound (a): "))
    b = float(input("Enter the initial upper bound (b): "))
    x, y = bounding_phase_method(a,b,question_no)
    if(x<a):
        x=a
    if(y>b):
        y=b
    x, y = interval_halving_method(x, y,question_no)

    print("The minimum point lies between  "+ str(round(x,4)) +" and " + str(round(y,4)))



if __name__ == "__main__":
    main_program()
