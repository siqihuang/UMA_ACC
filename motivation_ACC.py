#from multiprocessing import Pool
import cPickle as nedosol
import sys as sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from snapshot_platform_ACC import *

def start_experiment():
    ### log and output files
    #sniffylog=open('sniffy.log', 'w')
    #sys.stdout=sniffylog
    #sniffypic=open('sniffypic.dat', 'w')

    ### number of steps
    STEPS=10000
    ### introduce "structure constants"
    X_BOUND=10 #length of the interval environment
    Y_BOUND=10
    THRESHOLD=1./((X_BOUND+1)*(Y_BOUND+1.)) #learning threshold for Sniffy
    INITIAL_CHARGE=4*(X_BOUND+Y_BOUND) #initial charge equals perimeter

    # discretization parameter
    TIME_CONST=1./(INITIAL_CHARGE**2)
    # value parameters
    PLAY_DRIFT=3./4.
    FORAGE_DRIFT=3./4.
    PLAY_VALUE=INITIAL_CHARGE/4
    FORAGE_VALUE=INITIAL_CHARGE
    CONVERT=.3

    ### open a new experiment
    EX=Experiment(1)
    
    ### initialize experiment
    X_START=rand(X_BOUND+1) #pick initial position of agent
    Y_START=rand(Y_BOUND+1)
    X_PLAY=rand(X_BOUND+1) #pick initial position of playground
    Y_PLAY=rand(Y_BOUND+1)    
    OK=False
    while not OK:
        X_CHARGER=rand(X_BOUND+1) #pick initial position of charging station
        Y_CHARGER=rand(Y_BOUND+1)
        OK=(X_CHARGER-X_PLAY)**2+(Y_CHARGER-Y_PLAY)**2>9
    

    ### add agents
    SNIFFY=EX.add_agent('Sniffy',THRESHOLD)

    ### introduce actions
    SNIFFY.add_sensor('rt')
    SNIFFY.add_sensor('lt')
    SNIFFY.add_sensor('up')
    SNIFFY.add_sensor('dn')
    #
    # remove all but the pure actions and the "no-action"
    templist=SNIFFY._GENERALIZED_ACTIONS[:]
    #print templist
    for item in templist:
        if sum(x%2 for x in item)<3:
            SNIFFY._GENERALIZED_ACTIONS.remove(item)
    print SNIFFY._GENERALIZED_ACTIONS

    #
    ### ``mapping'' system
    #

    ### introduce agent's position, '(xpos,ypos)':
    def xmotion(state):
        xdiff=0
        if state['rt'][0] and state['xpos'][0]+1 in range(X_BOUND):
            xdiff+=1
        if state['lt'][0] and state['xpos'][0]-1 in range(X_BOUND):
            xdiff+=-1
	#print state['xpos'][0]+xdiff
        return state['xpos'][0]+xdiff

    def ymotion(state):
        ydiff=0
        if state['up'][0] and state['ypos'][0]+1 in range(Y_BOUND):
            ydiff+=1
        if state['dn'][0] and state['ypos'][0]-1 in range(Y_BOUND):
            ydiff+=-1
        return state['ypos'][0]+ydiff

    INIT=X_START
    EX.add_measurable('xpos',[INIT,INIT],xmotion)

    INIT=Y_START
    EX.add_measurable('ypos',[INIT,INIT],ymotion)
    
    # set up position sensors
    def xsensor(m):
        return lambda state: state['xpos'][0]<m+1
    #
    # setting up positional context for actions
    for ind in xrange(X_BOUND):
        tmp_name='x'+str(ind)
        SNIFFY.add_sensor(tmp_name,xsensor(ind))
        SNIFFY.twedge('rt',tmp_name)
        SNIFFY.twedge('rt',name_comp(tmp_name))
        SNIFFY.twedge('lt',tmp_name)
        SNIFFY.twedge('lt',name_comp(tmp_name))
    
    def ysensor(m):
        return lambda state: state['ypos'][0]<m+1
    #
    # setting up positional context for actions
    for ind in xrange(Y_BOUND):
        tmp_name='y'+str(ind)
        SNIFFY.add_sensor(tmp_name,ysensor(ind))
        SNIFFY.twedge('up',tmp_name)
        SNIFFY.twedge('up',name_comp(tmp_name))
        SNIFFY.twedge('dn',tmp_name)
        SNIFFY.twedge('dn',name_comp(tmp_name))


    #
    ### motivational system
    #

    # available charge
    def charge(state):
        if state['nav2'][0]<=2./float((X_BOUND+1)**2+(Y_BOUND+1)**2):
            return INITIAL_CHARGE
        if state['xpos'][0]!=state['xpos'][1] or state['ypos'][0]!=state['ypos'][1]: return state['charge'][0]-1
        else:
            if state['xpos'][0]==X_PLAY and state['ypos'][0]==Y_PLAY: return state['charge'][0]-2
            else: return state['charge'][0]

    INIT=INITIAL_CHARGE
    EX.add_measurable('charge',[INIT,INIT],charge)

    # normalized distance to playground (nav function #1)
    def nav1(state):
        return float((state['xpos'][0]-X_PLAY)**2+(state['ypos'][0]-Y_PLAY)**2)/float((X_BOUND+1)**2+(Y_BOUND+1)**2)

    INIT=nav1(EX.state_all())
    EX.add_measurable('nav1',[INIT,INIT],nav1)

    # normalized distance to charger (nav function #2)
    def nav2(state):
        return float((state['xpos'][0]-X_CHARGER)**2+(state['ypos'][0]-Y_CHARGER)**2)/float((X_BOUND+1)**2+(Y_BOUND+1)**2)

    INIT=nav2(EX.state_all())
    EX.add_measurable('nav2',[INIT,INIT],nav2)


    ### value dynamics
    # attractive value of play (note position, charge and navs have been updated!)
    def val1(state):
        return state['val1'][0]+TIME_CONST*PLAY_DRIFT*(PLAY_VALUE*state['nav1'][1]-state['val1'][0])

    # attractive value of foraging grows as charge diminishes (note position, charge and navs have already been updated!)
    def val2(state):
        return state['val2'][0]+TIME_CONST*FORAGE_DRIFT*((1+FORAGE_VALUE-state['charge'][1])*state['nav2'][1]-state['val2'][0])

    INIT=PLAY_VALUE
    EX.add_measurable('val1',[INIT,INIT],val1)

    INIT=FORAGE_VALUE
    EX.add_measurable('val2',[INIT,INIT],val2)

    ### value sensing
    def closer_to_playground(state):
        return state['nav1'][0]<state['nav1'][1]
    def closer_to_charger(state):
        return state['nav2'][0]<state['nav2'][1]
    
    SNIFFY.add_sensor('to_playground',closer_to_playground)
    SNIFFY.add_sensor('to_charger',closer_to_charger)

    SNIFFY.add_eval('to_playground')
    SNIFFY.add_eval('to_charger')

    
    ### motivation dynamics (vals have already been updated!)
    def mot1(state):
        m1=state['mot1'][0]+TIME_CONST*(-state['mot1'][0]/state['val1'][1]+state['val1'][1]*(1-state['mot1'][0]-state['mot2'][0])*(1+state['mot1'][0])-CONVERT*state['mot1'][0]*state['mot2'][0])
        m2=state['mot2'][0]+TIME_CONST*(-state['mot2'][0]/state['val2'][1]+state['val2'][1]*(1-state['mot2'][0]-state['mot1'][0])*(1+state['mot2'][0])-CONVERT*state['mot2'][0]*state['mot1'][0])

        return simplex_normalize([m1,m2,1-m1-m2])[0]
        #return m1

    INIT=1
    EX.add_measurable('mot1',[INIT,INIT],mot1)

    # taking into account that $mot1$ has already been updated
    def mot2(state):
        m1=state['mot1'][0]
        m2=state['mot2'][0]+TIME_CONST*(-state['mot2'][0]/state['val2'][1]+state['val2'][1]*(1-state['mot2'][0]-state['mot1'][1])*(1+state['mot2'][0])-CONVERT*state['mot2'][0]*state['mot1'][1])

        return simplex_normalize([m1,m2,1-m1-m2])[1]
        #return m2

    INIT=0
    EX.add_measurable('mot2',[INIT,INIT],mot2)

    ### Run
    #plt.ion()
    screen=plt.figure()

    X_locations=[]
    Y_locations=[]
    
    def update_step(iter):
        screen.clf()
        X_locations.append(EX.state('xpos')[0])
        Y_locations.append(EX.state('ypos')[0])

        ### create a histogram of position history
        heatmap,xedges,yedges=np.histogram2d(X_locations,Y_locations,range=[[0,X_BOUND],[0,Y_BOUND]],bins=[X_BOUND+1,Y_BOUND+1])
        extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]

        my_map=plt.imshow(heatmap.T,extent=extent,cmap='hot',animated=True) 
        ax=my_map.axes

        ax.plot(X_PLAY,Y_BOUND-Y_PLAY,'bo',X_CHARGER,Y_BOUND-Y_CHARGER,'b^',EX.state('xpos')[0],Y_BOUND-EX.state('ypos')[0],'b>',markersize=20)

        message=EX.tick([EX.state('mot1')[0],EX.state('mot2')[0],1-EX.state('mot1')[0]-EX.state('mot2')[0]])
        ax.set_title('charge='+str(EX.state('charge')[0])+', decision='+str(message))
        #print message
        return my_map

    #sniffypic.close()
    #sniffylog.close()

    ani=animation.FuncAnimation(screen,update_step,STEPS,blit=False)
    plt.show()

start_experiment()
exit(0)
