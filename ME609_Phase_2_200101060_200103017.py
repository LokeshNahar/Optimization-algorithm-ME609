
# from ME609_Phase_1_200101060_200103017 import bounding_phase_method, interval_halving_method
import math
import random
import numpy as np


def create_unit_vector(arr):
    # Calculate the magnitude of the vector
    magnitude = np.linalg.norm(arr)

    # Ensure the vector is not a zero vector
    if magnitude == 0:
        raise ValueError("Cannot create a unit vector from a zero vector.")

    # Create the unit vector by dividing each element by the magnitude
    unit_vector = arr / magnitude

    return unit_vector


def objective_function(x, question_no):
    n = len(x)
    ans = 0
    match question_no:
        case 1:
            # print("Sum Squares Function: ")
            for i in range(n):
                ans += (i+1) * pow(x[i], 2)
        case 2:
            # print("Rosenbrock Function: ")
            for i in range(n - 1):
                term1 = 100 * (x[i + 1] - x[i]**2)**2
                term2 = (1 - x[i])**2
                ans += term1 + term2
        case 3:
            # print("Dixons price Function: ")
            ans = (x[0] - 1)**2
            for i in range(2, n + 1):
                ans += i * (2 * x[i - 1]**2 - x[i - 2])**2

        case 4:
            # print("Trid Function: ")

            ans = sum((x[i] - 1)**2 for i in range(n)) - \
                sum(x[i] * x[i-1] for i in range(1, n))

        case 5:
            # print("Zakharov Function: ")

            term1 = sum(x[i]**2 for i in range(n))
            term2 = sum(0.5 * (i + 1) * x[i] for i in range(n))
            term3 = term2**2
            term4 = term2**4
            ans = term1 + term3 + term4

        case 6:
            return (pow((x[0]**2 + x[1] - 11), 2)+pow((x[0]+x[1]**2-7), 2))
    # print(ans)
    return ans


def calculate_distance(point1, point2):
    # Ensure the points are NumPy arrays for element-wise operations
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the squared differences between corresponding coordinates
    squared_diff = (point2 - point1)**2

    # Sum the squared differences and take the square root
    distance = np.sqrt(np.sum(squared_diff))

    return distance


# Interval Halving method
def interval_halving_method(a, b, current, e, question_no, epsilon=1e-3):

    # print("*********************************************")
    # print("Interval Halving Method")
    # epsilon = float(input("Enter the epsilon value(by default 10^-3): "))
    # value of xm
    x_m = (a+b)/2
    # length
    l = b-a
    # f(xm) function evaluation
    f_x_m = objective_function(current + x_m*e, question_no)
    # counter for function evaluations
    f_eval = 1
    # counter for num_iterations
    iter = 1
    # create a file according to the question/function number

    # out.write(str(iter)+"\t\t"+str(round(a,4))+"\t\t"+"nil"+"\t\t"+"nil"+"\t\t"+str(round(b,4))+"\t\t"+str(round(x_m,4))+"\t\t"+"nil"+"\t\t"+"nil"+"\t\t"+str(round(f_x_m,4)))
    # if |l|<epsilon   we break
    while (abs(l) >= epsilon):

        # step 2
        x_1 = a+(l/4)
        x_2 = b-(l/4)
        f_x_1 = objective_function(current + x_1*e, question_no)
        f_x_2 = objective_function(current + x_2*e, question_no)
        f_eval += 2
        # 2 more evaluations for each iterations
        # step 3
        if (f_x_1 < f_x_m):
            b = x_m
            x_m = x_1
            f_x_m = f_x_1
        elif (f_x_2 < f_x_m):
            a = x_m
            x_m = x_2
            f_x_m = f_x_2
        else:
            a = x_1
            b = x_2
        # step 4 new length
        l = b-a

        iter += 1
        # writting to the log file
        # out.write(str(iter)+"\t\t"+str(round(a,4))+"\t\t"+str(round(x_1,4))+"\t\t"+str(round(x_2,4))+"\t\t"+str(round(b,4))+"\t\t"+str(round(x_m,4))+"\t\t"+str(round(f_x_1,4))+"\t\t"+str(round(f_x_2,4))+"\t\t"+str(round(f_x_m,4)))

    # print("*********************************************")
    return a, b


def bounding_phase_method(current, e, i, question_no, delta=0.02):
    k = 0
    n = len(current)
    # step 1 set k = 0 , choose initial guess
    # print("*********************************************")
    # print("Bounding Phase Method")
    x_0 = 0.01
    while (1):
        # x_0 = float(input("Enter initial guess: "))
        x_0 = random.randint(int(-10), int(10)+1)
        # x_0 = 0.02
        f_x_0 = objective_function(current + x_0*e, question_no)
        f_x_0_del_minus = objective_function(
            current + ((x_0-delta)*e), question_no)
        f_x_0_del_plus = objective_function(
            current + ((x_0+delta)*e), question_no)
        if (f_x_0_del_minus <= f_x_0 <= f_x_0_del_plus):
            delta = -delta
            break
        elif (f_x_0_del_minus >= f_x_0 >= f_x_0_del_plus):
            break
        else:
            pass

    x_k = x_k_minus_1 = x_0
    x_k_plus_1 = x_k + ((2**k)*(delta))
    f_x_k_plus_1 = objective_function(current + x_k_plus_1*e, question_no)
    f_x_k = objective_function(current + x_k * e, question_no)

    f_eval = 2
    iter = 1

    while (f_x_k_plus_1 < f_x_k):
        k += 1
        iter += 1
        x_k_minus_1 = x_k
        x_k = x_k_plus_1
        x_k_plus_1 = x_k + ((2**k)*(delta))
        f_x_k = f_x_k_plus_1
        f_x_k_plus_1 = objective_function(current + x_k_plus_1*e, question_no)
        # out.write(str(iter)+"\t\t"+str(round(x_k_minus_1,4))+"\t\t"+str(round(x_k,4))+"\t\t"+str(round(x_k_plus_1,4))+"\t\t"+str(round(f_x_k,4))+"\t\t"+str(round(f_x_k_plus_1,4)))

        f_eval += 1

    if (x_k_minus_1 == min(x_k_minus_1, x_k_plus_1)):
        return x_k_minus_1, x_k_plus_1
    else:
        return x_k_plus_1, x_k_minus_1


def helper_function(current, e, i, question_no):
    a, b = bounding_phase_method(current, e, i, question_no)
    a, b = interval_halving_method(a, b, current, e, question_no)
    z = (a+b)/2
    return z


# Powell's Conjugate Direction Method
def powells_conjugate_direction(starting_position, question_no, epsilon=1e-4, max_iterations=1000):
    n = len(starting_position)
    # e = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    # directions=np.array(e)
    directions = np.eye(n)

    x = starting_position
    x1 = starting_position + 2 * 10
    num_iterations = 0
    # termination Condition
    # while (abs(directions[: ,n-1])>epsilon)
    while np.max(np.abs(x - x1)) > epsilon:
        num_iterations += 1
        current = x
        # first n searches for s1 s2 .... upto sn
        for i in range(n):
            alpha = helper_function(current, directions[:, i], i, question_no)
            current = current + alpha * directions[:, i]
            print(f"{objective_function(current,question_no)}")
        # new directions
        for i in range(n - 1):
            directions[:, i] = directions[:, i + 1]

        directions[:, n - 1] = create_unit_vector(current - x)

        # last search along s1 which is now curr - x ....(d)
        x1 = x
        alpha = helper_function(current, directions[:, n-1], n-1, question_no)
        x = current + alpha * directions[:, n - 1]
        # if number pf iterations exceed
        # print("Iteration no: ", num_iterations)

        if (num_iterations > max_iterations):
            break
# the Optimal function Value.
    fx = objective_function(x, question_no)

    return x, fx, num_iterations


def main_program():
    # question_no = int(input("Enter question number: "))

    # n = int(input("Enter number of variables: "))

    # ls = []
    # for _ in range(n):
    #     k = int(input())
    #     ls.append(k)
    # x = np.array(ls)
    file_path = "input.txt"

# Read question number and number of variables from the file
    with open(file_path, 'r') as file:
        question_no = int(file.readline().strip())
        n = int(file.readline().strip())

    # Read values from the file
    with open(file_path, 'r') as file:
        # Skip the first two lines (already read question_no and n)
        for _ in range(2):
            file.readline()

        # Read the next n lines for the variables
        ls = [int(file.readline().strip()) for _ in range(n)]

    x = np.array(ls)

    # e =np.array([1,0])
    # result = powells_conjugate_direction(x, question_no)
    # print("Optimal solution:", result)
    # print(objective_function(x,question_no))
    # a,b = bounding_phase_method(x,e,question_no)
    # a,b = interval_halving_method(a,b,question_no)
    optimal_vector,optimal_function_value,num_iterations = powells_conjugate_direction(x, question_no)
    out = open(r"Powell_Conjugate_direction_output_"+str(question_no)+".txt", "w")  # Output file
    out.write(f"Optimal Vector: {optimal_vector}\nThe optimal functional value is: {optimal_function_value}\nwith number of iterations: {num_iterations}")


    # print(a,b)


if __name__ == "__main__":
    main_program()
