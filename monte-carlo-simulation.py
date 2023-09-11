import numpy as np
import scipy.stats
import scipy.linalg as sc
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.stats import kde
from scipy.special import factorial
from scipy.stats import chisquare


"""
 run the code to see menu

"""
###########################################################################################################################################
#                                                             TASK 1                                                                      #
###########################################################################################################################################
def analytical(N):
    """
    analytical method for producing a sin distribution
    """
    tic = time.time() #time at start
    toc=np.zeros(int(N))
    theta=np.zeros(int(N))
    for i in range(int(N)):
        theta[i]=(np.arccos(np.random.uniform(-1.0,1.0))) # thetas given by arccos of random numbers between 0 and 1
        toc[i]=time.time() - tic # time after each iteration
    return theta,toc


def reject_accept(pdf,N,xmin,xmax):
      """
      reject accept method for producing a sin distribution
      """
      tic = time.time()
      toc=np.zeros(int(N)+1)
      x=np.linspace(xmin,xmax,int(N)) #range of random numbers needed to be generated
      y=pdf(x) # y is range of values pdf(x) can take
      pmin=0.
      pmax=y.max()  # max value of pdf(x)

      n_acc=0
      rand=[]
      while n_acc<N:
          x=np.random.uniform(xmin,xmax)  # generate random x
          y=np.random.uniform(pmin,pmax)  # generate uniform random y

          if y<pdf(x): #if inside pdf
              rand.append(x)
              toc[n_acc] = time.time() - tic
              n_acc+=1
      rand=np.asarray(rand)   # make array
      return rand,toc

def hist_t1(N,option):
    """
    histogram of analytical and accept rejects for sine function
    """
    theta=np.linspace(0,np.pi,N) # theta for sine function
    if option == 1: # analytical
        observed,_=analytical(N)
        plt.hist(observed, bins='auto',density=True, alpha=0.3,histtype='stepfilled', color='steelblue',edgecolor='k',label='Analytical method')
        plt.plot(theta,0.5*np.sin(theta),label='Sine function',color='r')
        plt.title('Histogram of the normalised frequency of theta, overlayed with a sin(theta) function.')
        plt.xlabel('Theta')
        plt.ylabel('Normalised Frequency')
        plt.legend(loc='best')
        plt.show()
    elif option == 0: #accept reject
        observed1,_=reject_accept(np.sin,N,0.0,np.pi)
        plt.hist(observed1, bins='auto',density=True, alpha=0.3,histtype='stepfilled', color='steelblue',edgecolor='k',label='Reject-Accept')
        plt.plot(theta,0.5*np.sin(theta),label='Sine function',color='r')
        plt.title('Histogram of the normalised frequency of theta, overlayed with a sin(theta) function.')
        plt.xlabel('Theta')
        plt.ylabel('Normalised Frequency')
        plt.legend(loc='best')
        plt.show()
    return

def error(N):
    """
    calculate absolute errors between sine and histograms
    """
    observed,_=analytical(N) # analytical
    observed1,_=reject_accept(np.sin,N,0.0,np.pi) #reject accept

    n,bins=np.histogram(observed,bins='auto',density=True)
    n2,bins2=np.histogram(observed1,bins='auto',density=True)

    expected=0.5*np.sin(bins) # expected, through normalisation the expected here is equal to normalised frequency of observed
    expected1=0.5*np.sin(bins2)

    error=(n-expected[:-1])**2 # sum of the squares
    error1=(n2-expected1[:-1])**2

    error_sum=np.sqrt(np.sum(error)) #modulus, analytical error
    error_sum1=np.sqrt(np.sum(error1)) #accept reject error

    return error_sum,error_sum1

def error_plot():
    """
    plots errors from error() function
    """
    N=np.linspace(10000,1000000,20) # range of samples

    error_sum=np.zeros(len(N)) # analytical
    error_sum1=np.zeros(len(N)) # accept reject

    for i in range(len(N)):
        error_sum[i],error_sum1[i] = error(N[i])
        print(i,'/20')

    plt.plot(N,1/(error_sum)**2,marker='o',label='Analytical') # should get straight line
    plt.plot(N,1/(error_sum1)**2,marker='o',label='Reject-Accept')
    plt.xlabel('Samples')
    plt.ylabel('Inverse of absolute error squared')
    plt.legend(loc='best')
    plt.title('Inverse of absolute error squared as a funtion of the number of samples.')
    plt.show()
    return

def time_task1(N):
    """
    Times analytical and accept reject
    """
    _,ta=analytical(N) # times of analytical
    _,tr=reject_accept(np.sin,N,0.0,np.pi)

    count=np.arange(0,len(ta),1) #samples analytical
    count2=np.arange(0,len(tr),1) #samples

    plt.plot(count[:-1],ta[:-1],label='Analytical')
    plt.plot(count2[:-1],tr[:-1],label='Reject-accept')
    plt.title('Time taken as a function of the number of samples, N.')
    plt.xlabel('Samples')
    plt.ylabel('Time(s)')
    plt.legend(loc='best')
    plt.show()
    return

def chi_square(N):
    """
    chi_square for comparison
    """
    theta_a,_=analytical(N)
    theta_r,_=reject_accept(np.sin,N,0.0,np.pi)

    obs_freq_a,bin_edges_a=np.histogram(theta_a,bins='auto',normed=True) # observed analytical, bin edges of histogram
    obs_freq_r,bin_edges_r=np.histogram(theta_r,bins='auto',normed=True)

    middle_a=np.zeros(len(obs_freq_a)) #middle analytical
    middle_r=np.zeros(len(obs_freq_r))

    for i in range(len(middle_a)):
        middle_a[i]=(bin_edges_a[i+1]-bin_edges_a[i])*0.5 #find middle values of each bin

    for i in range(len(middle_r)):
        middle_r[i]=(bin_edges_r[i+1]-bin_edges_r[i])*0.5

    exp_freq_a=0.5*np.sin(middle_a)
    exp_freq_r=0.5*np.sin(middle_r) #expected reject accept from middle values

    _,p_a=scipy.stats.chisquare(obs_freq_a,exp_freq_a) #p value of analytical chi squared
    _,p_r=scipy.stats.chisquare(obs_freq_r,exp_freq_r)

    print(middle_a)

    print("P-value for the analytical method at N =",N,"is P=",p_a)
    print("P-value for the analytical method at N =",N,"is P=",p_r)
    return

###########################################################################################################################################
#                                                       PHYSICS PROBLEM 1                                                                 #
###########################################################################################################################################
def decay(x):
    """
    function used for decay rate of nuclei
    """
    y=np.exp(-1.04*x)
    return y # return probability

def coordinates_at_detection(N,smear):
    """
    gives coordinates at detection of gamma ray
    """
    x_coords=np.array([])
    y_coords=np.array([])

    thetas,_=analytical(N) # theta in sine distribution
    phis=np.random.uniform(0,2*np.pi,N)

    randdist,_=reject_accept(decay,N,0,2) # gives a random distance between 0 and 2 in decay pdf
    det_dist=(2-randdist) # distance to detector at point of emission
    for i in range(N):
        if det_dist[i] <= 0: # goes beyond detector
            pass
        if thetas[i] >= np.pi/2: # reject values in lower half of sphere
            pass
        else:
            r=(det_dist[i])/np.cos(thetas[i]) # new radius of sphere to get coordinates
            new_x_coords=r*np.sin(thetas[i])*np.cos(phis[i]) # coordinates at detection
            new_y_coords=r*np.sin(thetas[i])*np.sin(phis[i])
            if abs(new_x_coords) > 1 or abs(new_y_coords) > 1: # limit to 1m by 1m detector
                pass
            else:
                if smear==1: # smear
                    smear_x=np.random.normal(new_x_coords,0.1/3)
                    smear_y=np.random.normal(new_y_coords,0.3/3)
                    if abs(smear_x) < 5 and abs(smear_y) < 5:
                        x_coords=np.append(x_coords,smear_x)
                        y_coords=np.append(y_coords,smear_y)
                elif smear ==0: # no smearing
                    x_coords=np.append(x_coords,new_x_coords)
                    y_coords=np.append(y_coords,new_y_coords)
    return x_coords,y_coords

def distribution_coords(N,smear):
    """
    plots distribution of coordinates
    """
    x_pixel=np.linspace(-1,1,100) # pixels for plotting, essentially 2d histogram
    y_pixel=np.linspace(-1,1,100)

    x,y=coordinates_at_detection(N,smear) # 1 is smear 0 is no smearing

    density=np.zeros([len(x_pixel)-1,len(y_pixel)-1])
    for i in range(len(x_pixel)-1):
        for j in range(len(y_pixel)-1):
            density[j,i]=((x_pixel[i] < x) & (x < x_pixel[i+1]) & (y_pixel[j] < y) & (y < y_pixel[j+1])).sum() # pixels of size 1/100 by 1/100 that count frequency of points inside

    plt.pcolormesh(x_pixel,y_pixel,density,cmap='hot')
    plt.colorbar(label='Frequency')
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title('Density plot of the distribution of gamma rays detected at a given coordinate (x,y).')
    plt.show()
    return


def test_spherical_dist(N):
    """
    test spherical distribution of coordinates is even
    """
    r=2 # radius of sphere
    theta=np.zeros(N)
    phi=np.zeros(N)

    x=np.zeros(N)
    y=np.zeros(N)
    z=np.zeros(N)

    theta,_=analytical(N) # let theta be sine distribution
    for i in range(N):
        phi[i]=np.random.uniform(0,2*np.pi)

        x[i] = r*np.sin(theta[i])*np.cos(phi[i]) # spherical polar coords
        y[i] = r*np.sin(theta[i])*np.sin(phi[i])
        z[i] = r*np.cos(theta[i])

    ax = Axes3D(plt.figure()) # 3d plot
    ax.scatter(x, y, z, c='blue',s=2,alpha=0.3)
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    ax.set_zlabel('z(m)')
    plt.show()
    return


def inverse_sq(N,smear):
    """
    tests inverse square law for detection points
    """
    x_pixel=np.linspace(-1,1,200) # just x pixels
    x,_=coordinates_at_detection(N,smear) # x coordinates of detection only

    densityx=np.zeros(len(x_pixel)) # frequency in each pixel

    for i in range(len(x_pixel)-1):
        densityx[i]=((x_pixel[i] < x) & (x < x_pixel[i+1])).sum() # sums the frequency inside this range of pixel

    plt.plot(x_pixel[:-1],np.sqrt(1/(densityx[:-1])))
    plt.xlabel('x(m)')
    plt.ylabel('Inverse of root of frequency')
    plt.title('Inverse of the root of the frequency against distance')
    plt.show()
    return

###########################################################################################################################################
#                                                       PHYSICS PROBLEM 2                                                                 #
###########################################################################################################################################
def total_count(sigma,N):
    """
    find the total count of background plus signal for a given cross section
    """
    background_mean=5.8

    mean_background=np.random.normal(background_mean,0.4,N) # normal distribution around mean of background, SD 0.4
    luminosity=np.random.normal(10,0.5,N) # normal dist of luminosity with SD 0.5

    background_count=np.random.poisson(mean_background)     #poisson dist for mean background
    signal_count=np.random.poisson(sigma*luminosity)    #poissson for signal

    total_count= background_count + signal_count #total is sum of two
    return total_count

def confidence(N):
    """
    finds sigma at which 95% of total count above 5
    """
    confidence=[]
    sigma=[]

    for i in range(N):
        sigma.append(0.001*i) # increase sigma
        total=total_count(sigma[i],N) # find total count at sigma

        above_five=sum(k > 5 for k in total) # sums number above 5
        percent=above_five/N # percentage above 5
        print(round(percent*100,1),'% confidence reached.')
        confidence.append(percent)
        if percent >= 0.95: # stop iterating when founf 95%
            plt.hist(total,bins='auto',alpha=0.3,histtype='stepfilled', color='steelblue',edgecolor='k') # plots histogram at this point
            plt.xlabel("Total counts")
            plt.ylabel("Frequency")
            plt.title("Frequency of a given total counts of gamma rays detected")
            plt.show()
            break
    plt.plot(sigma,confidence)
    plt.xlabel('Cross-section (sigma)')
    plt.ylabel('Confidence')
    plt.title("Confidence level of values above 5 counts for a given cross section")
    plt.show()
    print('Cross section at which the total counts are 95% above 5 =',round(np.max(sigma)),2)
    return

###########################################################################################################################################
#                                                                 MENU                                                                   #
###########################################################################################################################################

def test_task1():
    """
    Menu system for task 1
    """
    MyInput = '0'
    while MyInput != 'q':
        print('-------------------------------TASK 1 MENU--------------------------------------------------------------------------')
        print("[1]Test 1: Tests the Analytical or 'Reject-Accept' method for producing a sine distribution.")
        print("[2]Test 2: Compares the two methods.")
        print("--------------------------------------------------------------------------------------------------------------")
        MyInput = input('Type 1, 2 or q to return to MAIN MENU:')
        if MyInput == '1':
            print("\n######## TEST 1 #########")
            MyInput= input('Would you like to use the Analytical or Reject-Accept method?(A,R) - CASE SENSITIVE.')
            if MyInput=='A':
                hist_t1(100000,1)
            elif MyInput=='R':
                hist_t1(100000,0)
            print("\n##########################")
        elif MyInput == '2':
            print("\n######## TEST 2 #########")
            MyInput= input('Would you like to compare computational time or error?(T,E) - CASE SENSITIVE.')
            if MyInput=='T':
                time_task1(1000000)
            elif MyInput=='E':
                error_plot()
            print("")
            print("\n##########################")
        else:
            print("")
    print("")
    print('Returning to main menu....')
    return

def test_task2():
    """
    Menu system for task 2
    """
    MyInput = '0'
    while MyInput != 'q':
        print('-------------------------------PHYSICS PROBLEM 1 MENU--------------------------------------------------------------------------------------------------------')
        print("[1]Test 1: Tests the exponential decay of the activity of the source using the Reject-Accept method, between the source and detector.")
        print("[2]Test 2: Plots distributions of the coordinates at which the decayed gamma ray hit the detector, with the option incorporate the resolution of the detector.")
        print("[3]Test 3: Verifies that the distribution of points over a sphere of radius r is uniform, using the sine distribution from Task 1.")
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
        MyInput = input('Type 1, 2, 3 or q to return to MAIN MENU:')
        if MyInput == '1':
            print("\n######## TEST 1 #########")
            ran,_=reject_accept(decay,100000,0,2)
            plt.hist(ran,bins='auto',density=True)
            plt.xlabel('Distance (m)')
            plt.ylabel('Activity')
            plt.title("Activity of the sample as a function of distance from source." )
            plt.show()
            print("\n##########################")
        elif MyInput == '2':
            print("\n######## TEST 2 #########")
            MyInput= input('Would you like to incorporate the resolution of the detector, using a 3 sigma standard deviation of the error?(Y/N) - CASE SENSITIVE.')
            if MyInput=='Y':
                distribution_coords(100000,1)
            elif MyInput=='N':
                distribution_coords(200000,0)
            print("")
            print("\n##########################")
        elif MyInput == '3':
            print("\n######## TEST 3 #########")
            test_spherical_dist(10000)
            print("")
            print("\n##########################")
        else:
            print("")
    print("")
    print('Returning to main menu....')
    return




def main_menu():
    """
    Menu system for overall code.
    """
    MyInput = '0'
    while MyInput != 'q':
        print("")
        print('------------------------------- MAIN MENU -------------------------------------------------------------')
        print("[1]Option 1: Performs task 1.")
        print("[2]Option 2: Performs Physics Problem 1.")
        print("[3]Option 3: Performs Physics Problem 2.")
        print("-------------------------------------------------------------------------------------------------------")
        print("")
        MyInput = input('Select 1, 2, 3 or q to quit:')
        if MyInput == '1':
            print("\n######## TASK 1  SELECTED #########")
            test_task1()
        elif MyInput == '2':
            print("\n######## PHYSICS PROBLEM 1 SELECTED #########")
            test_task2()
        elif MyInput == '3':
            print("\n######## PHYSICS PROBLEM 2 SELECTED #########")
            confidence(30000)
        else:
            print("")
    print('Goodbye')
    return

chi_square(500)
