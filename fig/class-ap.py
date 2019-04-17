# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 22:49:10 2017

@author: wzmsltw
"""

import matplotlib.pyplot as plt
import numpy as np

fig_size = (8, 6)

names = ['SSAD', 'Decouple-SSAD (UCF101)', 'Decouple-SSAD (KnetV3)']
subjects = [' BaseballPitch', 'BasketballDunk', '     Billiards', '  CleanAndJerk',
            '   CliffDiving', 'CricketBowling', '   CricketShot', '        Diving',
            '  FrisbeeCatch', '     GolfSwing', '   HammerThrow', '      HighJump',
            '  JavelinThrow', '      LongJump', '     PoleVault', '       Shotput',
            ' SoccerPenalty', '   TennisSwing', '   ThrowDiscus', 'VolleyballSpiking', '           mAP']

scores_1 = [29.3, 9.2, 4.7, 35.6,
            46.1, 10.0, 1.9, 17.6,
            6.6, 13.3, 51.6, 21.6,
            42.7, 71.3, 58.1, 21.8,
            13.3, 8.8, 24.8, 3.9, 24.6]
scores_2 = [17.3, 23.6, 0.65, 37.3,
            55.6, 18.4, 1.9, 53.1,
            10.2, 36.9, 69.0, 49.9,
            62.4, 78.9, 80.9, 24.3,
            5.7, 5.8, 59.4, 11.1, 35.4]
scores_3 = [32.1, 28.3, 06.1, 48.8,
            59.7, 16.0, 4.1, 58.3,
            19.7, 39.6, 74.5, 73.4,
            82.2, 94.5, 82.4, 39.6,
            10.0, 9.2, 70.5, 14.6, 43.2]

scores = [scores_1, scores_2, scores_3]

font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 12,
        }

bar_width = 0.2

index = np.arange(len(scores[0]))

rects1 = plt.bar(index, scores[0], bar_width, color='gray', label=names[0])

rects2 = plt.bar(index + bar_width, scores[1], bar_width, color='yellow', label=names[1])

rects3 = plt.bar(index + 2 * bar_width, scores[2], bar_width, color='orange', label=names[2])

plt.xticks(index - 0.3, subjects, rotation=30, fontsize=14, family='serif')

plt.ylim(ymax=100, ymin=0)
plt.axis([-0.5, 21, 0, 100])
plt.tight_layout()

plt.legend(loc='lower right', bbox_to_anchor=(1, 0.7), fancybox=True, ncol=1)
plt.ylabel("Average Presicion(%)", fontdict=font)

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')

        rect.set_edgecolor('white')

# add_labels(rects1)
# add_labels(rects2)
# add_labels(rects3)
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()
plt.savefig('compare_ap.png')
