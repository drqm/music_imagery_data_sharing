from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
from psychopy import gui, visual, event, core
from os import system, path, chdir
from random import shuffle

my_path = path.abspath(path.dirname(__file__))
chdir(my_path)

def quit():
    win0.close()
    core.quit()

event.globalKeys.add(key='escape', func=quit, name='shutdown')

ID_box = gui.Dlg(title = 'Subject identity')
ID_box.addField('ID: ')
ID_box.addField('tasks to include, separated by commas:\n'
                '(all tasks: Leave blank; learningBach: 1; localizer: 2; manipulation: 3)')

sub_id = ID_box.show()

tasks = ["learningBachShort_iEEG.py",
         "localizer_iEEG.py",
         "manipulation_task_iEEG.py"]

task_order = [[1],[2,3]]
shuffle(task_order)
task_order = task_order[0] + task_order[1]

if sub_id[1]:
   to = sub_id[1].split(',')
   task_order = [int(tt) for tt in to]

print('current task order is: ' + str(task_order))

for tsk_ix, tsk in enumerate(task_order):
    chdir(my_path)
    system('python ' + tasks[tsk-1] + ' ' + sub_id[0])
    if tsk_ix < len(task_order) - 1:
        win0 = visual.Window(fullscr=True, color='black')
        wait_scr = visual.TextStim(win0, text = "Now take a little break and rest as much as you need.\n"
                                                "Press a key to continue with the next task when ready.",
                                                wrapWidth=1.8, color = 'white')
        wait_scr.draw()
        win0.flip()
        event.waitKeys()
        win0.close()
