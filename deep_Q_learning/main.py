import deep_q_learning as Taxi
from deep_q_learning import DQN as model_class
from test_model import TestModel


def train(pt_path=None):
    taxi = Taxi.QAgent(pt_path)
    taxi.compile()
    taxi.fit()
    return taxi.pt_path

def test(pt_path):
    model = TestModel(model_class, pt_path, render_mode='rgb_array')
    model.test(test_episodes=5, # number of test episode to execute
               timestamp=0.1, # time between each frame
               fast_testing=False, # display graphical interface or print only test informations
               final_frame_pause=1) # time to wait after the last frame of each episode

if __name__ ==  "__main__":
    cmd = str(input('Do you want to train a new model ? (y/n)\n'))
    if cmd == 'y':
        #? TRAIN NEW MODEL
        pt_path = train()
        cmd = str(input('Do you want to test this model ? (y/n)\n'))
        if cmd == 'y':
            #* TEST NEW MODEL
            print('\nLoading model from file: ' + pt_path)
            res = None if input('Continue ? (y/n)\n') == 'y' else exit(0)
            test(pt_path)
    else:
        cmd = str(input('Do you want to use another model ? (y/n)\n'))
        if cmd == 'y':
            # LOAD MODEL FROM FILE
            pt_path = 'model_backup/' + str(input('Enter the path to the model file:\n'))
            print('\nLoading model from file: ' + pt_path)
            cmd = str(input('Do you wish to test or to resume training ? (test/resume)\n'))
            if cmd == 'test':
                #* TEST LOADED MODEL
                test(pt_path)
            elif cmd == 'resume':
                #? RESUME TRAINING
                train(pt_path)
                cmd = str(input('Do you want to test this model ? (y/n)\n'))
                if cmd == 'y':
                    #* TEST RESUMED MODEL
                    print('\nLoading model from file: ' + pt_path)
                    res = None if input('Continue ? (y/n)\n') == 'y' else exit(0)
                    test(pt_path)
