# -*- coding: utf-8 -*-
"""
Based on code shared by wzmsltw
"""

import matplotlib.pyplot as plt
import numpy as np

fig_size = (8, 6)

names = ['SSAD (UCF101)', 'Decouple-SSAD (UCF101)', 'Decouple-SSAD (KnetV3)']
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
scores_2 = [22.4, 21.0, 6.0, 36.5,
            55.3, 18.2, 2.6, 53.1,
            9.8, 36.3, 68.8, 52.0,
            65.8, 78.2, 80.3, 31.1,
            8.5, 4.9, 56.8, 8.6, 35.8]
scores_3 = [33.2, 28.3, 7.6, 48.1,
            56.9, 14.0, 5.5, 58.8,
            11.9, 42.1, 75.2, 76.6,
            83.8, 94.6, 84.0, 40.8,
            15.1, 9.3, 71.4, 6.8, 43.7]

scores = [scores_1, scores_2, scores_3]

font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 10,
        }

bar_width = 0.2

index = np.arange(len(scores[0]))

rects1 = plt.bar(index, scores[0], bar_width, color='gray', label=names[0])

rects2 = plt.bar(index + bar_width, scores[1], bar_width, color='pink', label=names[1])

rects3 = plt.bar(index + 2 * bar_width, scores[2], bar_width, color='orange', label=names[2])

plt.xticks(index - 0.3, subjects, rotation=30, fontsize=10, family='serif')

plt.ylim(ymax=100, ymin=0)
plt.axis([-0.5, 21, 0, 100])
plt.tight_layout()

plt.legend(loc='lower right', bbox_to_anchor=(1, 0.7), fancybox=True, ncol=1)
plt.ylabel("AP@0.5 of THUMOS14 (%)", fontdict=font)

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
# plt.savefig('compare_ap.png')
