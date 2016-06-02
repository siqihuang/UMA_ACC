#import sys as sys
import numpy as np
import curses


from snapshot_platform import *

def start_experiment(stdscr):
    ### log and output files
    #sniffylog=open('sniffy.log', 'w')
    #sys.stdout=sniffylog
    #sniffypic=open('sniffypic.dat', 'w')

    ### introduce "structure constants"
    DRY_RUN_CYCLES=1
    X_BOUND=10 #length of the interval environment
    Y_BOUND=10
    THRESHOLD=1./((X_BOUND+1)*(Y_BOUND+1.)) #learning threshold for Sniffy

    ### open a new experiment
    EX=Experiment(1)
    
    ### initialize experiment
    X_START=rand(X_BOUND+1) #pick initial position of agent
    Y_START=rand(Y_BOUND+1)
    OK=False
    while not OK:
        X_PLAY=rand(X_BOUND+1) #pick initial position of charging station
        Y_PLAY=rand(Y_BOUND+1)
        OK=((X_START-X_PLAY)**2+(Y_START-Y_PLAY)**2>9)
    

    ### add agents
    SNIFFY=EX.add_agent('Sniffy',THRESHOLD)

    ### introduce actions
    EX.new_sensor([SNIFFY],'rt')
    EX.new_sensor([SNIFFY],'lt')
    EX.new_sensor([SNIFFY],'up')
    EX.new_sensor([SNIFFY],'dn')
    
    # remove all but the pure actions and the "no-action"
    templist=SNIFFY._GENERALIZED_ACTIONS[:]
    for item in templist:
        if sum(x%2 for x in item)<3:
            SNIFFY._GENERALIZED_ACTIONS.remove(item)

    # naming the available generalized actions for future use
    RIGHT=[0,3,5,7]
    LEFT=[1,2,5,7]
    UP=[1,3,4,7]
    DOWN=[1,3,5,6]
    STAND_STILL=[1,3,5,7]


    #
    ### ``mapping'' system
    #

    ### introduce agent's position, '(xpos,ypos)':
    def xmotion(state):
        xdiff=0
        if state['rt'][0] and state['xpos'][0]+1 in range(X_BOUND+1):
            xdiff+=1
        if state['lt'][0] and state['xpos'][0]-1 in range(X_BOUND+1):
            xdiff+=-1
        return state['xpos'][0]+xdiff

    def ymotion(state):
        ydiff=0
        if state['up'][0] and state['ypos'][0]+1 in range(Y_BOUND+1):
            ydiff+=1
        if state['dn'][0] and state['ypos'][0]-1 in range(Y_BOUND+1):
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
        EX.new_sensor([SNIFFY],tmp_name,xsensor(ind))
        EX.twedge([SNIFFY],'rt',tmp_name)
        EX.twedge([SNIFFY],'rt',name_comp(tmp_name))
        EX.twedge([SNIFFY],'lt',tmp_name)
        EX.twedge([SNIFFY],'lt',name_comp(tmp_name))

    def ysensor(m):
        return lambda state: state['ypos'][0]<m+1
    #
    # setting up positional context for actions
    for ind in xrange(Y_BOUND):
        tmp_name='y'+str(ind)
        EX.new_sensor([SNIFFY],tmp_name,ysensor(ind))
        EX.twedge([SNIFFY],'up',tmp_name)
        EX.twedge([SNIFFY],'up',name_comp(tmp_name))
        EX.twedge([SNIFFY],'dn',tmp_name)
        EX.twedge([SNIFFY],'dn',name_comp(tmp_name))


    #
    ### motivational system
    #

    # normalized distance to playground (nav function #1)
    def nav(state):
        return np.fabs(state['xpos'][0]-X_PLAY)+np.fabs(state['ypos'][0]-Y_PLAY)

    INIT=nav(EX.state_all())
    EX.add_measurable('nav',[INIT,INIT],nav)

    ### value sensing
    def closer_to_playground(state):
        return state['nav'][0]<state['nav'][1]
    
    EX.new_sensor([SNIFFY],'to_playground',closer_to_playground)
    SNIFFY.add_eval('to_playground')


    ### Run
    
    # prepare windows for output
    curses.curs_set(0)
    stdscr.erase()
    WIN=curses.newwin(Y_BOUND+3,X_BOUND+3,6,7)
    stdscr.nodelay(1)
    WIN.border(int(35),int(35),int(35),int(35),int(35),int(35),int(35),int(35))
    WIN.bkgdset(int(46))
    WIN.overlay(stdscr)
    WIN.noutrefresh()

    # output subroutine
    def print_state(counter,text):
        if stdscr.getch()==int(32):
            raise('Aborting at your request...\n\n')
        stdscr.clear()
        stdscr.addstr('S-N-I-F-F-Y  I-S  R-U-N-N-I-N-G    (press [space] to stop) ')
        stdscr.addstr(4,3,text)
        stdscr.clrtoeol()
        stdscr.noutrefresh()
        WIN.clear()
        WIN.addstr(0,0,str(counter))
        WIN.addch(Y_BOUND+1-Y_PLAY,1+X_PLAY,int(84)) # print target (playground)
        WIN.addch(Y_BOUND+1-EX.state('ypos')[0],1+EX.state('xpos')[0],int(83)) # print sniffy's position
        WIN.overlay(stdscr)
        WIN.noutrefresh()
        curses.doupdate()
    #print SNIFFY._SIZE
    acc.initData(SNIFFY._SIZE,THRESHOLD,SNIFFY._CONTEXT.keys(),SNIFFY._CONTEXT.values())
    

    # SETTING UP DRY RUN
    count=-(2*(X_START+Y_START)+DRY_RUN_CYCLES*(4*X_BOUND+4*Y_BOUND))
    message='DRY RUN'
    print_state(count,message)

    # DRY RUN : GO TO ORIGIN
    if X_START>0:
        for ind in xrange(X_START):
            message='DRY RUN: '+EX.tick('execute',LEFT)
            print_state(count,message)
            count+=1
    if Y_START>0:
        for ind in xrange(Y_START):
            message='DRY RUN: '+EX.tick('execute',DOWN)
            print_state(count,message)
            count+=1

    # DRY RUN : CYCLE ALONG THE RIM
    for cycle in xrange(DRY_RUN_CYCLES):
        # COUNTER-CLOCKWISE:
        for ind in xrange(X_BOUND):
            message='DRY RUN: '+EX.tick('execute',RIGHT)
            print_state(count,message)
            count+=1
        for ind in xrange(Y_BOUND):
            message='DRY RUN: '+EX.tick('execute',UP)
            print_state(count,message)
            count+=1
        for ind in xrange(X_BOUND):
            message='DRY RUN: '+EX.tick('execute',LEFT)
            print_state(count,message)
            count+=1
        for ind in xrange(Y_BOUND):
            message='DRY RUN: '+EX.tick('execute',DOWN)
            print_state(count,message)
            count+=1
        # CLOCKWISE:
        for ind in xrange(Y_BOUND):
            message='DRY RUN: '+EX.tick('execute',UP)
            print_state(count,message)
            count+=1
        for ind in xrange(X_BOUND):
            message='DRY RUN: '+EX.tick('execute',RIGHT)
            print_state(count,message)
            count+=1
        for ind in xrange(Y_BOUND):
            message='DRY RUN: '+EX.tick('execute',DOWN)
            print_state(count,message)
            count+=1
        for ind in xrange(X_BOUND):
            message='DRY RUN: '+EX.tick('execute',LEFT)
            print_state(count,message)
            count+=1

    # DRY RUN : RETURN TO STARTING POSITION
    for ind in xrange(X_START):
        message='DRY RUN: '+EX.tick('execute',RIGHT)
        print_state(count,message)
        count+=1
    for ind in xrange(Y_START):
        message='DRY RUN: '+EX.tick('execute',UP)
        print_state(count,message)
        count+=1

    # REAL RUN : GO TO TARGET
    while stdscr.getch()!=int(32):
        print_state(count,message)
        message='RUNNING: '+EX.tick('decide','to_playground')
        count+=1
    #print SNIFFY._SIZE
    
curses.wrapper(start_experiment)
exit(0)
