# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:20:19 2019

@author: Admin
"""

import eikon as ek
ek.set_app_key('88728d9ceb094a57aaa16f1bdceb8d899dafab9a')
p1 = ek.get_timeseries(["JNJ.N"], start_date = "2010-02-01T15:04:05", end_date = "2017-02-05T15:04:05", interval="tick")