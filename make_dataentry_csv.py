#!/usr/bin/env python3
"""
Build a long-format data-entry CSV from a results.xlsx produced by survey_ocr.py.

One row per (survey, question) for ALL 29 questions. MCQ answers are filled with
the FULL option text as printed on the form; handwritten questions are left blank
for you to type in. A 'Needs entry/review' column flags what still needs a human.

Usage:  python3 make_dataentry_csv.py results.xlsx  [output.csv]
"""
import sys, csv
from openpyxl import load_workbook

# Full question wording, transcribed from the printed form.
QUESTIONS = {
 "Q1":  ("What bus route did you receive this survey? (Route #)", "text"),
 "Q2":  ("What time did you board this bus? (time + AM/PM)",      "text"),
 "Q3":  ("Where did you come from today?",                        "Q3 - Came From"),
 "Q4":  ("What is the address of the place you came from? (Street / City / State / Zip)", "text"),
 "Q5":  ("How did you get to this bus? (choose primary method)",  "Q5 - To Bus"),
 "Q6":  ("Where did you get ON this bus? (terminal/bus stop)",    "text"),
 "Q7":  ("Where will you get OFF this bus? (terminal/bus stop)",  "text"),
 "Q8":  ("After this bus, how will you get to your final destination?", "Q8 - After Bus"),
 "Q9":  ("What is the address of your final destination?",        "text"),
 "Q10": ("Where are you going today?",                            "Q10 - Going To"),
 "Q11": ("How often do you use this bus route?",                  "Q11 - Frequency"),
 "Q12": ("What type of ticket are you using for this trip?",      "Q12 - Ticket"),
 "Q13": ("Do you typically pay for your pass using?",             "Q13 - Pay"),
 "Q14": ("Where do you usually buy your ticket or pass?",         "Q14 - Buy Ticket"),
 "Q15": ("For the other half of your trip (return trip), how will/did you travel?", "Q15 - Return"),
 "Q16": ("What is the purpose of your trip?",                     "Q16 - Trip Purpose"),
 "Q17": ("If this bus service was not available, how would you make the trip?", "Q17 - Alt Mode"),
 "Q18": ("How do you identify?",                                  "Q18 - Gender"),
 "Q19": ("What is your age?",                                     "text"),
 "Q20": ("Are you traveling with a child(ren) under 12 years of age?", "Q20 - Traveling With Child"),
 "Q21": ("How well do you speak English?",                        "Q21 - English"),
 "Q22": ("Do you speak a language other than English?",           "Q22 - Other Language"),
 "Q23": ("Are you of Spanish/Hispanic/Latino origin?",            "Q23 - Hispanic/Latino"),
 "Q24": ("What is your race?",                                    "Q24 - Race"),
 "Q25": ("How many people are living in your household, including yourself?", "text"),
 "Q26": ("How many vehicles are currently available in your household?", "text"),
 "Q27": ("Do you have a physical disability or other condition that makes it difficult to use this bus?", "Q27 - Disability"),
 "Q28": ("What is your annual household income?",                 "Q28 - Income"),
 "Q29": ("What is the most important bus improvement that can be made to meet your travel needs?", "text"),
}

# Map the tool's short option label -> full option text on the form.
OPTIONS = {
 "Q3 - Came From": {"Home":"Home","Work":"Work","School":"School","Other":"Other (please specify)"},
 "Q5 - To Bus": {"Walked Only":"Walked Only","Bike":"Bike","Scooter":"Scooter","Drive":"Drive","Taxi":"Taxi",
   "Carpooled/Dropped Off":"Carpooled/Dropped Off","Uber/Lyft":"Uber/Lyft/Other rideshare",
   "Another Bus":"Another Bus","Riverline":"Riverline","Light Rail":"Light Rail",
   "NJ Transit Train":"NJ Transit Train","SEPTA":"SEPTA"},
 "Q8 - After Bus": {"Walked Only":"Walked Only","Bike":"Bike","Scooter":"Scooter","Drive":"Drive","Taxi":"Taxi",
   "Carpooled":"Carpooled/Dropped Off","Uber/Lyft":"Uber/Lyft/Other rideshare",
   "Another Bus":"Another Bus","Riverline":"Riverline","Light Rail":"Light Rail",
   "NJ Transit Train":"NJ Transit Train","SEPTA":"SEPTA"},
 "Q10 - Going To": {"Home":"Home","Work":"Work","School":"School","Other":"Other (please specify)"},
 "Q11 - Frequency": {"Less than once/year":"less than once year","1-5 times/year":"1-5 times a year",
   "6-11 times/year":"6-11 times a year","1-3 days/month":"1-3 days a month","1-2 days/week":"1-2 days per week",
   "2 days/week":"2 days per week","3 days/week":"3 days a week","4 days/week":"4 days a week",
   "5 days/week":"5 days a week","6-7 days/week":"6-7 days a week"},
 "Q12 - Ticket": {"One-way/Cash":"One-way/Cash Fare/Transfer","Bus Monthly Pass":"Bus Monthly Pass",
   "Rail Monthly Pass":"Rail Monthly Pass","College Student Monthly":"College Student Monthly Pass",
   "Student Ticket":"Student Ticket (one-way and transfers)","Ten-Trip":"Ten-Trip",
   "Reduced Fares (Senior/Disability)":"Reduced Fares for Senior Citizens & Customers with Disabilities"},
 "Q13 - Pay": {"Cash":"Cash","Google Pay":"Google Pay","Cash from Transit Wallet":"Cash from Transit Wallet",
   "Debit Card":"Debit Card","Credit Card":"Credit Card","Check":"Check","Apple Pay":"Apple Pay",
   "PayPal":"Paypal","Other":"Other (please specify)"},
 "Q14 - Buy Ticket": {"On-board":"On-board","NJ Transit Mobile App":"NJ Transit Mobile App",
   "Ticket Vending Machine":"Ticket Vending Machine","NJ TRANSIT Ticket Agent":"NJ TRANSIT Ticket agent",
   "Independent Agent":"Independent ticket agent (e.g., newsstand, convenience store)","Other":"Other (please specify)"},
 "Q15 - Return": {"Will Not Make":"I will not make this trip in the opposite direction",
   "Same Bus 6am-8:59am":"Same Bus Route 6am to 8:59am","Same Bus 9am-3:59pm":"Same Bus Route 9am to 3:59pm",
   "Same Bus 4pm-6:59pm":"Same Bus Route 4pm to 6:59pm","Same Bus 7pm-9:59pm":"Same Bus Route 7pm to 9:59pm",
   "Same Bus 10pm-11:59pm":"Same Bus Route 10pm to 11:59pm","Same Bus midnight-5:59am":"Same Bus Route midnight to 5:59am",
   "Drive":"Drive","Taxi":"Taxi","Uber/Lyft":"Uber/Lyft/Other rideshare","Another Bus":"Another Bus",
   "PATCO":"PATCO","NJ Transit Train":"NJ Transit Train","SEPTA":"SEPTA","Other":"Other (please specify)"},
 "Q16 - Trip Purpose": {"Work":"Work","Company Business":"Company business (not your normal commute)",
   "School":"School","Shopping":"Shopping","Entertainment":"Entertainment/Recreational","Medical":"Medical",
   "Social/Family":"Social/ visiting family and friends","Personal Business":"Personal Business"},
 "Q17 - Alt Mode": {"Would Not Make Trip":"I would not make this trip","Walk":"Walk","Bike":"Bike",
   "Scooter":"Scooter","Drive":"Drive","Taxi":"Taxi","Carpooled":"Carpooled/Dropped Off",
   "Uber/Lyft":"Uber/Lyft/Other rideshare","Other":"Other (please specify)"},
 "Q18 - Gender": {"Female/Woman":"Female/Woman","Male/Man":"Male/Man",
   "Non-Binary/Gender Fluid":"Non-Binary/Gender Fluid","Prefer Not to Answer":"Prefer not to answer"},
 "Q20 - Traveling With Child": {"Yes":"Yes","No":"No"},
 "Q21 - English": {"Very Well":"Very well","Well":"Well","Not Well":"Not well","Not At All":"Not at all"},
 "Q22 - Other Language": {"No":"No","Yes":"Yes (please specify)"},
 "Q23 - Hispanic/Latino": {"No":"No","Yes":"Yes (please specify)"},
 "Q24 - Race": {"White/Caucasian":"White/Caucasian","Asian/Pacific Islander":"Asian or Pacific Islander",
   "Mixed Race":"Mixed Race","Black/African American":"Black or African American",
   "American Indian/Alaskan Native":"American Indian/Alaskan Native","Prefer Not To Answer":"Prefer not to answer"},
 "Q27 - Disability": {"No":"No","Vision":"Yes, a disability affecting my vision",
   "Hearing":"Yes, a disability affecting my hearing","Mobility":"Yes, a disability affecting my mobility",
   "Other":"Yes, a disability not listed above"},
 "Q28 - Income": {"Under $15k":"Under $15,000","$15k-$24.9k":"$15,000-$24,999","$25k-$34.9k":"$25,000-$34,999",
   "$35k-$49.9k":"$35,000-$49,999","$50k-$74.9k":"$50,000-$74,999","$75k-$99.9k":"$75,000-$99,999",
   "$100k-$149.9k":"$100,000-$149,999","$150k-$199.9k":"$150,000-$199,999","$200k-$249.9k":"$200,000-$249,999",
   "$250k+":"$250,000 or more"},
}


def main():
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "results.xlsx"
    out  = sys.argv[2] if len(sys.argv) > 2 else xlsx.rsplit(".", 1)[0] + "_dataentry.csv"
    wb = load_workbook(xlsx)
    ws = wb["Survey Results"]
    hdr = [c.value for c in ws[1]]
    surveys = hdr[1:]
    # read the group->value grid
    grid = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0]:
            grid[row[0]] = {surveys[i]: row[1 + i] for i in range(len(surveys))}
    # read the needs-review grid (same layout) if present
    review = {}
    if "Needs Review" in wb.sheetnames:
        rv = wb["Needs Review"]
        for row in rv.iter_rows(min_row=2, values_only=True):
            if row[0]:
                review[row[0]] = {surveys[i]: (row[1 + i] not in (None, ""))
                                  for i in range(len(surveys))}

    # Wide layout: one row per question, one column per survey.
    rows = []
    for qn, (qtext, kind) in QUESTIONS.items():
        row = [qn, qtext]
        for sv in surveys:
            if kind == "text":
                row.append("")                       # handwritten — not auto-read
            else:
                raw = (grid.get(kind, {}) or {}).get(sv)
                row.append(OPTIONS.get(kind, {}).get(raw, raw) if raw else "")
        rows.append(row)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Q#", "Question"] + list(surveys))
        w.writerows(rows)
    print(f"Wrote {len(rows)} question rows x {len(surveys)} survey columns -> {out}")


if __name__ == "__main__":
    main()
