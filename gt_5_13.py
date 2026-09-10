import json
# Manually-read answers normalized to config option labels
GT = {
 '0480': {'Q2 - Board Time':None,'Q3 - Came From':'Home','Q5 - To Bus':'Walked Only','Q8 - After Bus':'Another Bus','Q10 - Going To':'Other','Q11 - Frequency':None,'Q12 - Ticket':'Bus Monthly Pass','Q13 - Pay':'Other','Q14 - Buy Ticket':'Other','Q15 - Return':'Another Bus','Q16 - Trip Purpose':'Medical','Q17 - Alt Mode':'Would Not Make Trip','Q18 - Gender':'Female/Woman','Q20 - Traveling With Child':'No','Q21 - English':'Very Well','Q22 - Other Language':'No','Q23 - Hispanic/Latino':'No','Q24 - Race':'White/Caucasian','Q27 - Disability':'No','Q28 - Income':'Under $15k'},
 '0603': {'Q2 - Board Time':None,'Q3 - Came From':'Other','Q5 - To Bus':'Walked Only','Q8 - After Bus':'Walked Only','Q10 - Going To':'Home','Q11 - Frequency':'6-7 days/week','Q12 - Ticket':'Bus Monthly Pass','Q13 - Pay':'Other','Q14 - Buy Ticket':'Other','Q15 - Return':'Same Bus 4pm-6:59pm','Q16 - Trip Purpose':'Shopping','Q17 - Alt Mode':'Uber/Lyft','Q18 - Gender':'Male/Man','Q20 - Traveling With Child':'No','Q21 - English':'Very Well','Q22 - Other Language':'No','Q23 - Hispanic/Latino':'No','Q24 - Race':'American Indian/Alaskan Native','Q27 - Disability':'No','Q28 - Income':'Under $15k'},
 '2011': {'Q2 - Board Time':None,'Q3 - Came From':'Work','Q5 - To Bus':'Walked Only','Q8 - After Bus':'Walked Only','Q10 - Going To':'Home','Q11 - Frequency':'6-7 days/week','Q12 - Ticket':'Bus Monthly Pass','Q13 - Pay':'Debit Card','Q14 - Buy Ticket':'NJ Transit Mobile App','Q15 - Return':'Same Bus 9am-3:59pm','Q16 - Trip Purpose':'Work','Q17 - Alt Mode':'Taxi','Q18 - Gender':'Female/Woman','Q20 - Traveling With Child':None,'Q21 - English':'Very Well','Q22 - Other Language':'No','Q23 - Hispanic/Latino':'No','Q24 - Race':'White/Caucasian','Q27 - Disability':'No','Q28 - Income':'$50k-$74.9k'},
 '2580': {'Q2 - Board Time':None,'Q3 - Came From':'Home','Q5 - To Bus':'Another Bus','Q8 - After Bus':'Walked Only','Q10 - Going To':'Other','Q11 - Frequency':'6-7 days/week','Q12 - Ticket':'Bus Monthly Pass','Q13 - Pay':'Other','Q14 - Buy Ticket':'Other','Q15 - Return':'Same Bus 9am-3:59pm','Q16 - Trip Purpose':'Medical','Q17 - Alt Mode':'Would Not Make Trip','Q18 - Gender':'Female/Woman','Q20 - Traveling With Child':'No','Q21 - English':'Very Well','Q22 - Other Language':'No','Q23 - Hispanic/Latino':'No','Q24 - Race':'White/Caucasian','Q27 - Disability':'No','Q28 - Income':'Under $15k'},
}
PEN = {'0480':'black pen/pencil','0603':'pencil','2011':'sky blue','2580':'black pen'}
json.dump({'answers':GT,'pen':PEN},open('answer_key_5_13.json','w'),indent=1)

# validate labels exist in config
cfg=json.load(open('form_config.json')); labels=set()
for f in cfg['fields']:
    if f.get('type')=='checkbox' and ': ' in f['name']: labels.add(f['name'].split(': ',1)[1])
bad=set()
for sv in GT.values():
    for q,a in sv.items():
        if a and a not in labels: bad.add(a)
print('labels NOT matching config (need fixing):', bad if bad else 'none — all match')
