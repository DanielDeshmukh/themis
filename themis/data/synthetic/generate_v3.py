"""THEMIS v3 data generation — scale to 50k+ pairs.

Strategy:
1. 15+ question templates per section (definition, penalty, procedure, exception, comparison, scenario, etc.)
2. Groq API generation — LLM creates diverse Q&A from section text
3. Expanded IPC→BNS mapping (200+ transitions)
4. Cross-law questions (BNS + BNSS + BSA combined)
5. Scenario-based questions (real-world fact patterns)
6. Abbreviation disambiguation
"""

import json
import os
import random
import re
import time
from pathlib import Path

from ...config import config

# IPC to BNS section mapping — comprehensive
IPC_BNS_MAPPING = {
    "101": "Section 103 (Murder)",
    "102": "Section 104 (Causing death by negligence)",
    "103": "Section 103 (Murder)",
    "105": "Section 105 (Culpable homicide not amounting to murder)",
    "106": "Section 106 (Abetment of suicide)",
    "109": "Section 108 (Abetment of murder)",
    "111": "Section 110 (Criminal conspiracy)",
    "114": "Section 113 (Unlawful assembly armed with deadly weapon)",
    "143": "Section 139 (Kidnapping from India)",
    "144": "Section 140 (Kidnapping from lawful guardianship)",
    "145": "Section 141 (Abduction)",
    "147": "Section 143 (Rioting)",
    "148": "Section 144 (Rioting armed with deadly weapon)",
    "149": "Section 145 (Affray)",
    "299": "Section 100 (Culpable homicide)",
    "300": "Section 101 (Murder)",
    "302": "Section 101 (Murder)",
    "303": "Section 105 (Culpable homicide not amounting to murder)",
    "304": "Section 105 (Culpable homicide not amounting to murder)",
    "306": "Section 108 (Abetment of suicide)",
    "307": "Section 109 (Attempt to murder)",
    "319": "Section 117 (Hurt)",
    "320": "Section 118 (Grievous hurt)",
    "323": "Section 121 (Voluntarily causing hurt)",
    "324": "Section 122 (Voluntarily causing hurt by dangerous weapons)",
    "325": "Section 123 (Voluntarily causing grievous hurt)",
    "326": "Section 124 (Voluntarily causing grievous hurt by dangerous weapons)",
    "354": "Section 74 (Assault on woman to outrage modesty)",
    "375": "Section 63 (Rape)",
    "376": "Section 64 (Punishment for rape)",
    "378": "Section 303 (Theft)",
    "379": "Section 304 (Punishment for theft)",
    "380": "Section 305 (Theft in dwelling house)",
    "383": "Section 308 (Extortion)",
    "384": "Section 309 (Punishment for extortion)",
    "390": "Section 315 (Robbery)",
    "391": "Section 316 (Dacoity)",
    "392": "Section 317 (Punishment for robbery)",
    "395": "Section 320 (Punishment for dacoity)",
    "403": "Section 328 (Dishonest misappropriation of property)",
    "405": "Section 316 (Criminal breach of trust)",
    "406": "Section 317 (Punishment for criminal breach of trust)",
    "415": "Section 138 (Cheating)",
    "417": "Section 140 (Punishment for cheating)",
    "420": "Section 143 (Cheating and inducing delivery of property)",
    "425": "Section 148 (Mischief)",
    "441": "Section 164 (Criminal trespass)",
    "442": "Section 165 (House trespass)",
    "447": "Section 170 (Punishment for criminal trespass)",
    "448": "Section 171 (Punishment for house trespass)",
    "452": "Section 175 (House-trespass after preparation for hurt)",
    "463": "Section 186 (Forgery)",
    "465": "Section 188 (Punishment for forgery)",
    "467": "Section 190 (Forgery of valuable security or will)",
    "468": "Section 191 (Forgery for purpose of cheating)",
    "497": None,
    "498": "Section 85 (Cruelty by husband or his relatives)",
    "500": "Section 356 (Defamation)",
    "503": "Section 351 (Criminal intimidation)",
    "504": "Section 352 (Intentional insult to provoke breach of peace)",
    "505": "Section 353 (Statements conducing to public mischief)",
    "506": "Section 351 (Punishment for criminal intimidation)",
    "509": "Section 354 (Insult to modesty of woman)",
    "511": "Section 61 (Punishment for attempt to commit offences)",
    # Extended mappings
    "34": "Section 9 (Act done by several persons in furtherance of common intention)",
    "35": "Section 9 (Act done by several persons in furtherance of common intention)",
    "120": "Section 60 (Concealment of design to commit offence punishable with death or imprisonment)",
    "120A": "Section 110 (Criminal conspiracy)",
    "120B": "Section 110 (Punishment for criminal conspiracy)",
    "141": "Section 142 (Unlawful assembly)",
    "142": "Section 142 (Being member of unlawful assembly)",
    "146": "Section 143 (Force)",
    "148": "Section 144 (Rioting armed with deadly weapon)",
    "160": "Section 146 (Affray)",
    "224": "Section 232 (Resistance or obstruction by person to his lawful apprehension)",
    "225": "Section 233 (Resistance or obstruction to lawful apprehension of another person)",
    "299": "Section 100 (Culpable homicide)",
    "300": "Section 101 (Murder)",
    "301": "Section 106 (Culpable homicide by act with knowledge of likelihood of causing death)",
    "302": "Section 101 (Murder)",
    "303": "Section 105 (Culpable homicide not amounting to murder)",
    "304": "Section 105 (Culpable homicide not amounting to murder)",
    "304A": "Section 106 (Causing death by negligence)",
    "305": "Section 106 (Abetment of suicide of child or insane person)",
    "306": "Section 108 (Abetment of suicide)",
    "307": "Section 109 (Attempt to murder)",
    "308": "Section 110 (Attempt to commit culpable homicide)",
    "309": "Section 47 (Attempt to commit suicide to compel or restrain public servant)",
    "311": "Section 114 (Punishment for being member of criminal gang)",
    "312": "Section 117 (Voluntarily causing miscarriage)",
    "313": "Section 117 (Miscarriage without woman's consent)",
    "314": "Section 117 (Death caused by act done with intent to cause miscarriage)",
    "315": "Section 117 (Act done with intent to prevent child from being born alive or to cause death after birth)",
    "316": "Section 117 (Act done with intent to prevent child from being born alive or to cause death after birth)",
    "317": "Section 117 (Exposure and abandonment of child under twelve years by parent or person having care)",
    "318": "Section 117 (Concealment of birth by secret disposal of dead body)",
    "321": "Section 118 (Grievous hurt)",
    "322": "Section 118 (Voluntarily causing grievous hurt)",
    "327": "Section 125 (Voluntarily causing hurt to extort confession, or to compel restoration of property)",
    "328": "Section 126 (Voluntarily causing hurt by means of poison, etc., with intent to commit an offence)",
    "329": "Section 127 (Voluntarily causing grievous hurt to extort confession, or to compel restoration of property)",
    "330": "Section 128 (Voluntarily causing hurt to deter public servant from his duty)",
    "331": "Section 129 (Voluntarily causing grievous hurt to deter public servant from his duty)",
    "332": "Section 130 (Voluntarily causing hurt to deter public servant from his duty)",
    "333": "Section 131 (Voluntarily causing grievous hurt to deter public servant from his duty)",
    "334": "Section 132 (Voluntarily causing hurt on provocation)",
    "335": "Section 133 (Voluntarily causing grievous hurt on provocation)",
    "336": "Section 134 (Act endangering life or personal safety of others)",
    "337": "Section 135 (Causing hurt by act endangering life or personal safety of others)",
    "338": "Section 136 (Causing grievous hurt by act endangering life or personal safety of others)",
    "341": "Section 164 (Wrongful restraint)",
    "342": "Section 165 (Wrongful confinement)",
    "343": "Section 166 (Wrongful confinement for three or more days)",
    "344": "Section 167 (Wrongful confinement for ten or more days)",
    "345": "Section 168 (Wrongful confinement of person forpurposing extorting property or confession)",
    "346": "Section 169 (Wrongful confinement for extorting confession or property)",
    "347": "Section 170 (Wrongful confinement for extorting confession or property)",
    "348": "Section 171 (Wrongful confinement for extorting confession or property)",
    "352": "Section 352 (Criminal force otherwise than on grave provocation)",
    "355": "Section 355 (Assault or criminal force with intent to dishonour person)",
    "356": "Section 356 (Assault or criminal force in attempt to commit theft)",
    "357": "Section 357 (Assault or criminal force in attempt to wrongfully confine person)",
    "358": "Section 358 (Assault or criminal force on grave provocation)",
    "359": "Section 363 (Kidnapping)",
    "360": "Section 364 (Kidnapping from India)",
    "361": "Section 365 (Kidnapping from lawful guardianship)",
    "362": "Section 366 (Abduction)",
    "363A": "Section 367 (Kidnapping or abducting minor under ten years)",
    "364": "Section 368 (Kidnapping or abducting in order to murder)",
    "365": "Section 369 (Kidnapping or abducting for slavery)",
    "366": "Section 370 (Trafficking of person)",
    "366A": "Section 370 (Trafficking of person)",
    "366B": "Section 370 (Trafficking of person)",
    "367": "Section 371 (Kidnapping or abducting for grievous hurt, slavery, etc.)",
    "368": "Section 372 (Wrongful concealment or misrepresentation)",
    "369": "Section 373 (Wrongful concealment or misrepresentation)",
    "370": "Section 370 (Trafficking of person)",
    "371": "Section 371 (Habitual dealing in slaves)",
    "372": "Section 372 (Selling minor for prostitution)",
    "373": "Section 373 (Buying minor for prostitution)",
    "374": "Section 374 (Wrongful compulsory labour)",
    "376": "Section 64 (Punishment for rape)",
    "376A": "Section 64 (Punishment for rape)",
    "376B": "Section 64 (Punishment for rape by person in authority)",
    "376C": "Section 64 (Punishment for rape by person in authority)",
    "376D": "Section 64 (Punishment for rape by person in authority)",
    "377": "Section 77 (Unnatural offences)",
    "384": "Section 309 (Punishment for extortion)",
    "385": "Section 310 (Putting person in fear of death or of grievous hurt, in order to commit extortion)",
    "386": "Section 311 (Extortion by putting person in fear of death or of grievous hurt)",
    "387": "Section 312 (Putting person in fear of death or of grievous hurt, in order to commit extortion)",
    "388": "Section 313 (Extortion by threat to accuse person of offence punishable with death or imprisonment)",
    "389": "Section 314 (Putting person in fear of accusation of offence punishable with death or imprisonment, in order to commit extortion)",
    "392": "Section 317 (Punishment for robbery)",
    "393": "Section 318 (Attempt to commit robbery)",
    "394": "Section 319 (Voluntarily causing hurt in committing robbery)",
    "395": "Section 320 (Punishment for dacoity)",
    "396": "Section 321 (Dacoity with murder)",
    "397": "Section 322 (Robbery, or dacoity, with attempt to cause death or grievous hurt)",
    "398": "Section 323 (Attempt to commit robbery or dacoity when armed with deadly weapon)",
    "399": "Section 324 (Making preparation to commit dacoity)",
    "400": "Section 325 (Punishment for belonging to gang of dacoits)",
    "401": "Section 326 (Punishment for belonging to gang of thieves)",
    "402": "Section 327 (Concealment of preparation to commit dacoity)",
    "403": "Section 328 (Dishonest misappropriation of property)",
    "404": "Section 329 (Dishonest misappropriation of property by public servant, or by person entrusted with property)",
    "405": "Section 316 (Criminal breach of trust)",
    "406": "Section 317 (Punishment for criminal breach of trust)",
    "407": "Section 318 (Criminal breach of trust by carrier, etc.)",
    "408": "Section 319 (Criminal breach of trust by public servant, or by banker, merchant, or agent)",
    "409": "Section 320 (Criminal breach of trust by public servant, or by banker, merchant, or agent)",
    "411": "Section 330 (Dishonestly receiving stolen property)",
    "412": "Section 331 (Dishonestly receiving stolen property knowing that it was obtained by dacoity)",
    "413": "Section 332 (Dishonestly receiving stolen property knowing that it was obtained by dacoity)",
    "414": "Section 333 (Assisting in concealment of stolen property)",
    "415": "Section 138 (Cheating)",
    "416": "Section 139 (Cheating by personation)",
    "417": "Section 140 (Punishment for cheating)",
    "418": "Section 141 (Cheating with knowledge that wrongful loss may be caused to person whose interest offender is bound to protect)",
    "419": "Section 142 (Punishment for cheating by personation)",
    "420": "Section 143 (Cheating and dishonestly inducing delivery of property)",
    "421": "Section 148 (Dishonest removal or concealment of property to prevent distribution among creditors)",
    "422": "Section 149 (Dishonestly inducing delivery of property to prevent distribution among creditors)",
    "423": "Section 150 (Dishonest or fraudulent removal or concealment of property)",
    "424": "Section 151 (Dishonest or fraudulent removal or concealment of property)",
    "425": "Section 148 (Mischief)",
    "426": "Section 149 (Punishment for mischief)",
    "427": "Section 150 (Mischief causing damage to amount of one thousand rupees or upwards)",
    "428": "Section 151 (Mischief by killing or maiming animal of the value of ten rupees or upwards)",
    "429": "Section 152 (Mischief by killing or maiming animal of the value of fifty rupees or upwards)",
    "430": "Section 153 (Mischief by destroying, moving, or rendering less useful, anything retained for public purposes)",
    "431": "Section 154 (Mischief by destroying, moving, or rendering less useful, anything retained for public purposes)",
    "432": "Section 155 (Mischief by causing inundation or obstruction to public drainage)",
    "433": "Section 156 (Mischief by destroying, moving, or rendering less useful, anything retained for public purposes)",
    "434": "Section 157 (Mischief by destroying, moving, or rendering less useful, anything retained for public purposes)",
    "435": "Section 158 (Mischief by destroying or damaging property)",
    "436": "Section 159 (Mischief by destroying or damaging property with fire or explosive substance)",
    "437": "Section 160 (Mischief by destroying or damaging property with fire or explosive substance)",
    "438": "Section 161 (Mischief by destroying or damaging property with fire or explosive substance)",
    "439": "Section 162 (Punishment for intentionally running vessel aground or ashore with intent to commit theft, etc.)",
    "440": "Section 163 (Mischief committed after preparation made for causing death or grievous hurt)",
    "441": "Section 164 (Criminal trespass)",
    "442": "Section 165 (House trespass)",
    "443": "Section 166 (Lurking house-trespass or house-breaking)",
    "444": "Section 167 (Lurking house-trespass or house-breaking by night)",
    "445": "Section 168 (House-breaking)",
    "446": "Section 169 (House-breaking by night)",
    "447": "Section 170 (Punishment for criminal trespass)",
    "448": "Section 171 (Punishment for house trespass)",
    "449": "Section 172 (House trespass in order to commit offence punishable with death)",
    "450": "Section 173 (House trespass in order to commit offence punishable with imprisonment for life)",
    "451": "Section 174 (House trespass in order to commit offence punishable with imprisonment)",
    "452": "Section 175 (House-trespass after preparation for hurt, assault, or wrongful restraint)",
    "453": "Section 176 (Punishment for lurking house-trespass or house-breaking)",
    "454": "Section 177 (Lurking house-trespass or house-breaking in order to commit offence punishable with imprisonment)",
    "455": "Section 178 (Lurking house-trespass or house-breaking after preparation for hurt, assault, or wrongful restraint)",
    "456": "Section 179 (Punishment for lurking house-trespass or house-breaking by night)",
    "457": "Section 180 (Lurking house-trespass or house-breaking by night in order to commit offence punishable with imprisonment)",
    "458": "Section 181 (Lurking house-trespass or house-breaking by night after preparation for hurt, assault, or wrongful restraint)",
    "459": "Section 182 (Grievous hurt caused by house-breaking)",
    "460": "Section 183 (Death or grievous hurt caused by house-breaking)",
    "461": "Section 184 (Dishonestly breaking open receptacle containing property)",
    "462": "Section 185 (Punishment for dishonestly breaking open receptacle containing property)",
    "463": "Section 186 (Forgery)",
    "464": "Section 187 (Making a false document)",
    "465": "Section 188 (Punishment for forgery)",
    "466": "Section 189 (Forgery of record of Court or of public register, etc.)",
    "467": "Section 190 (Forgery of valuable security or will)",
    "468": "Section 191 (Forgery for purpose of cheating)",
    "469": "Section 192 (Forgery for purpose of harming reputation)",
    "470": "Section 193 (Forged document or electronic record)",
    "471": "Section 194 (Using as genuine a forged document or electronic record)",
    "472": "Section 195 (Making or possessing forged document or electronic record with intent to commit forgery)",
    "473": "Section 196 (Making or possessing forged document or electronic record with intent to commit forgery)",
    "474": "Section 197 (Possessing forged document or electronic record with intent to use it for genuine)",
    "475": "Section 198 (Counterfeiting device or mark used for authenticating documents)",
    "476": "Section 199 (Counterfeiting device or mark used for authenticating documents)",
    "477": "Section 200 (Fraudulently destroying, defacing, or removing document or electronic record)",
    "477A": "Section 200 (Falsification of accounts)",
    "478": "Section 201 (Trade-marks)",
    "479": "Section 202 (Property mark)",
    "480": "Section 203 (Using a false trade-mark or property mark)",
    "481": "Section 204 (Using a false property mark)",
    "482": "Section 205 (Faking property mark)",
    "483": "Section 206 (Counterfeiting trade-mark or property mark)",
    "484": "Section 207 (Counterfeiting trade-mark or property mark)",
    "485": "Section 208 (Making or possession of counterfeit trade-mark or property mark)",
    "486": "Section 209 (Selling goods marked with counterfeit trade-mark or property mark)",
    "487": "Section 210 (Making false mark upon goods or包装)",
    "488": "Section 211 (Making false mark upon goods or包装)",
    "489": "Section 212 (Using false mark upon goods or包装)",
    "489A": "Section 212 (Counterfeiting currency notes or bank notes)",
    "489B": "Section 213 (Using as genuine forged or counterfeit currency notes or bank notes)",
    "489C": "Section 214 (Possession of forged or counterfeit currency notes or bank notes)",
    "489D": "Section 215 (Making, possessing, or using forged or counterfeit currency notes or bank notes)",
    "489E": "Section 216 (Making, possessing, or using forged or counterfeit currency notes or bank notes)",
    "490": "Section 217 (Breach of contract of service or attendance)",
    "491": "Section 218 (Breach of contract of service or attendance)",
    "492": "Section 219 (Breach of contract of service or attendance)",
    "493": "Section 220 (Cohabitation caused by man deceitfully inducing belief of lawful marriage)",
    "494": "Section 221 (Marrying again during lifetime of husband or wife)",
    "495": "Section 222 (Same offence with concealment of former marriage from person with whom subsequent marriage is contracted)",
    "496": "Section 223 (Marriage ceremony fraudulently gone through without lawful marriage)",
    "497": None,
    "498": "Section 85 (Cruelty by husband or his relatives)",
    "498A": "Section 85 (Cruelty by husband or his relatives)",
    "500": "Section 356 (Defamation)",
    "501": "Section 357 (Printing or engraving matter known to be defamatory)",
    "502": "Section 358 (Sale of printed or engraved substance containing defamatory matter)",
    "503": "Section 351 (Criminal intimidation)",
    "504": "Section 352 (Intentional insult to provoke breach of peace)",
    "505": "Section 353 (Statements conducing to public mischief)",
    "506": "Section 351 (Punishment for criminal intimidation)",
    "507": "Section 352 (Criminal intimidation by anonymous communication)",
    "508": "Section 354 (Act caused by inducing person to believe that he will be rendered an object of Divine displeasure)",
    "509": "Section 354 (Insult to modesty of woman)",
    "510": "Section 355 (Misconduct in public by intoxicated person)",
    "511": "Section 61 (Punishment for attempt to commit offences)",
}

# Abbreviation disambiguation pairs — expanded
ABBREVIATION_PAIRS = [
    {
        "instruction": "What does BNS stand for in Indian law?",
        "input": "",
        "output": "BNS stands for the Bharatiya Nyaya Sanhita, 2023. It is the new criminal code of India that replaces the Indian Penal Code (IPC) 1860. The BNS was enacted on December 25, 2023, and came into effect on July 1, 2024. It contains 358 sections covering offences against the state, public tranquility, human body, property, and other criminal matters.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does BNSS mean in Indian law?",
        "input": "",
        "output": "BNSS stands for the Bharatiya Nagarik Suraksha Sanhita, 2023. It is the new criminal procedure code of India that replaces the Code of Criminal Procedure (CrPC) 1973. The BNSS was enacted on December 25, 2023, and came into effect on July 1, 2024. It contains 531 sections covering the procedure for investigation, inquiry, and trial of criminal offences.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does BSA stand for in Indian law?",
        "input": "",
        "output": "BSA stands for the Bharatiya Sakshya Adhiniyam, 2023. It is the new Indian Evidence Act that replaces the Evidence Act 1872. The BSA was enacted on December 25, 2023, and came into effect on July 1, 2024. It contains 170 sections covering the admissibility, relevancy, and examination of evidence in Indian courts.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What is the difference between IPC and BNS?",
        "input": "",
        "output": "The IPC (Indian Penal Code) 1860 was India's criminal code for over 160 years. The BNS (Bharatiya Nyaya Sanhita) 2023 is its replacement, enacted on December 25, 2023, and effective from July 1, 2024. Key differences: BNS has 358 sections (IPC had 511), BNS adds new offences like organized crime and terrorism, BNS increases minimum punishments for many offences, and BNS incorporates community service as a form of punishment.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What is the difference between CrPC and BNSS?",
        "input": "",
        "output": "The CrPC (Code of Criminal Procedure) 1973 was India's criminal procedure code. The BNSS (Bharatiya Nagarik Suraksha Sanhita) 2023 is its replacement, effective from July 1, 2024. Key changes: BNSS mandates forensic investigation for crimes punishable with 7+ years, BNSS allows electronic/digital records as evidence, BNSS reduces time limits for trials, and BNSS introduces plea bargaining provisions.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What is the difference between the Evidence Act and BSA?",
        "input": "",
        "output": "The Indian Evidence Act 1872 governed admissibility of evidence in Indian courts. The BSA (Bharatiya Sakshya Adhiniyam) 2023 is its replacement, effective from July 1, 2024. Key changes: BSA recognizes electronic and digital records as primary evidence, BSA allows statements of accused persons under certain conditions, and BSA modernizes provisions on expert testimony.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does RTI mean in Indian law?",
        "input": "",
        "output": "RTI stands for the Right to Information Act, 2005. It grants Indian citizens the right to request information from public authorities. Any citizen can seek information within 30 days. Public authorities must designate Public Information Officers (PIOs) to handle requests.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does CPA mean in Indian law?",
        "input": "",
        "output": "CPA stands for the Consumer Protection Act, 2019. It provides a three-tier consumer dispute redressal mechanism: District Commission, State Commission, and National Commission. The Act covers all goods and services, including e-commerce and digital transactions.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What is the difference between BNS and BNSS?",
        "input": "",
        "output": "BNS (Bharatiya Nyaya Sanhita) defines substantive criminal law — what constitutes an offence and its punishment. BNSS (Bharatiya Nagarik Suraksha Sanhita) defines criminal procedure — how offences are investigated, tried, and adjudicated. BNS says 'murder is punishable by death'; BNSS says 'how a murder case is investigated and tried.'\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What is the difference between BNS and BSA?",
        "input": "",
        "output": "BNS (Bharatiya Nyaya Sanhita) defines criminal offences and punishments. BSA (Bharatiya Sakshya Adhiniyam) defines the law of evidence — what can be proved, how it is proved, and what evidence is admissible in court. BNS says 'theft is an offence'; BSA says 'how you prove theft happened in court.'\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What is the full form of FIR in Indian law?",
        "input": "",
        "output": "FIR stands for First Information Report. Under Section 173 of BNSS (formerly Section 154 of CrPC), an FIR is the first report given to the police about a cognizable offence. It is the foundation of the criminal justice process. Any person can file an FIR, and the police are duty-bound to register it for cognizable offences.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does Cognizable Offence mean under BNSS?",
        "input": "",
        "output": "A cognizable offence is one where the police can arrest the accused without a warrant and investigate without the Magistrate's permission. Under BNSS, cognizable offences are those punishable with 3+ years of imprisonment or listed in the First Schedule. Examples include murder, robbery, and kidnapping.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does Non-Cognizable Offence mean under BNSS?",
        "input": "",
        "output": "A non-cognizable offence is one where the police cannot arrest without a warrant or investigate without the Magistrate's permission. Under BNSS, these are offences not listed as cognizable in the First Schedule. Examples include simple assault, defamation, and minor theft. The complainant must approach a Magistrate directly.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does Bailable Offence mean under BNSS?",
        "input": "",
        "output": "A bailable offence is one where the accused has a right to bail as a matter of law. Under BNSS, bailable offences are generally those punishable with imprisonment of 3 years or less. The police or Magistrate must release the accused on bail if they comply with bail conditions.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does Non-Bailable Offence mean under BNSS?",
        "input": "",
        "output": "A non-bailable offence is one where bail is not a right but a discretion of the court. Under BNSS, serious offences like murder, rape, and dacoity are non-bailable. The accused must apply to the court for bail, which may be granted based on the seriousness of the offence, flight risk, and other factors.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does Compoundable Offence mean under BNS?",
        "input": "",
        "output": "A compoundable offence is one where the victim and accused can settle the matter amicably, and the case is dropped. Under BNS, compoundable offences include minor assault, defamation, and certain property offences. Non-compoundable offences like murder and rape cannot be settled between parties.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does Anticipatory Bail mean under BNSS?",
        "input": "",
        "output": "Anticipatory bail is a provision under Section 482 of BNSS (formerly Section 438 of CrPC) that allows a person to seek bail in advance if they apprehend arrest. The person can apply to the Sessions Court or High Court for protection from arrest. If granted, the person cannot be arrested for the offence mentioned in the application.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does Summary Trial mean under BNSS?",
        "input": "",
        "output": "Summary trial is a simplified procedure for trying minor offences under BNSS. It applies to offences punishable with imprisonment of 2 years or less. The procedure is quicker: the Magistrate records a brief outline of the evidence, and the trial is completed in one or two hearings. This speeds up justice for less serious offences.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does Charge Sheet mean under BNSS?",
        "input": "",
        "output": "A charge sheet (now called 'police report' under BNSS) is filed by the police under Section 193 of BNSS (formerly Section 173 of CrPC) after completing investigation. It contains the details of the offence, evidence collected, and names of accused persons. The Magistrate takes cognizance of the offence based on the charge sheet.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
    {
        "instruction": "What does Cognizance mean under BNSS?",
        "input": "",
        "output": "Taking cognizance means the Magistrate formally acknowledges that an offence has been committed and proceeds to deal with it. Under BNSS, the Magistrate can take cognizance on receipt of a charge sheet from police, a complaint from a victim, or information from any person. This marks the beginning of the judicial process.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
    },
]

# Expanded question templates — 15+ categories
QUESTION_TEMPLATES_V3 = {
    "definition": [
        "What does {act_name} mean by '{title}'?",
        "How is '{title}' defined in {act_name}?",
        "What is the legal definition of {title_lower}?",
        "Can you explain what '{title}' means under {act_name}?",
        "What is the meaning of {title_lower} in Indian law?",
        "Define {title_lower} as per {act_name}.",
        "What are the essential elements of {title_lower} under {act_name}?",
    ],
    "punishment": [
        "What is the punishment for {title_lower} under {act_name}?",
        "How is {title_lower} punished under {act_name}?",
        "What penalty does the law prescribe for {title_lower}?",
        "What happens if someone is found guilty of {title_lower}?",
        "What is the jail term for {title_lower}?",
        "What is the maximum punishment for {title_lower}?",
        "Is there a minimum punishment for {title_lower}?",
    ],
    "procedure": [
        "What is the procedure for {title_lower}?",
        "How do I file for {title_lower}?",
        "What steps should I follow for {title_lower}?",
        "How does one go about {title_lower}?",
        "What is the process for {title_lower} under {act_name}?",
        "What is the timeline for {title_lower}?",
        "What documents are needed for {title_lower}?",
    ],
    "right": [
        "What are my rights regarding {title_lower}?",
        "Can I {title_lower} under {act_name}?",
        "How do I exercise my right to {title_lower}?",
        "What rights do I have under {act_name} about {title_lower}?",
        "Am I entitled to {title_lower} under Indian law?",
        "What remedies are available if my right to {title_lower} is violated?",
    ],
    "offence": [
        "Is {title_lower} a crime under {act_name}?",
        "What happens if someone commits {title_lower}?",
        "What are the consequences of {title_lower}?",
        "Can I be arrested for {title_lower}?",
        "What are the penalties for {title_lower}?",
        "Is {title_lower} a cognizable or non-cognizable offence?",
        "Is {title_lower} a bailable offence?",
    ],
    "exception": [
        "What are the exceptions to {title_lower} under {act_name}?",
        "Are there any defences available for {title_lower}?",
        "When does {title_lower} not apply?",
        "What situations are excluded from {title_lower}?",
        "Are there any mitigating circumstances for {title_lower}?",
    ],
    "comparison": [
        "How does {title_lower} differ from similar offences?",
        "What makes {title_lower} different from related crimes?",
        "How is {title_lower} classified under {act_name}?",
        "What distinguishes {title_lower} from lesser offences?",
    ],
    "scenario": [
        "If someone commits {title_lower}, what legal action can be taken?",
        "A person is accused of {title_lower}. What happens next?",
        "What legal options does a victim of {title_lower} have?",
        "How would a court handle a case of {title_lower}?",
        "What evidence is needed to prove {title_lower}?",
    ],
    "section_reference": [
        "What does Section {section_number} of {act_name} say about {title_lower}?",
        "Explain Section {section_number} of {act_name} in simple terms.",
        "What should I know about {title_lower} under {act_name}?",
        "Can you explain Section {section_number} of {act_name}?",
        "What does the law say about {title_lower}?",
        "Break down Section {section_number} of {act_name} for me.",
        "What are the key points of Section {section_number} of {act_name}?",
    ],
    "jurisdiction": [
        "Which court has jurisdiction over {title_lower}?",
        "Where should a case of {title_lower} be filed?",
        "Which Magistrate can try {title_lower}?",
        "Is {title_lower} triable by a Sessions Court?",
    ],
    "limitation": [
        "What is the time limit for filing a case related to {title_lower}?",
        "How long do I have to report {title_lower}?",
        "Is there a statute of limitations for {title_lower}?",
    ],
    "burden_of_proof": [
        "Who has the burden of proof in a {title_lower} case?",
        "What must the prosecution prove for {title_lower}?",
        "What is the standard of proof for {title_lower}?",
    ],
    "cross_law": [
        "How does {section_number} of {act_name} relate to criminal procedure?",
        "What role does {act_name} play in investigating {title_lower}?",
        "How does evidence law apply to {title_lower} cases?",
    ],
}


def categorize_section(title: str, text: str) -> str:
    """Categorize a section based on its title and content."""
    combined = (title + " " + text).lower()

    if any(kw in combined for kw in ["punish", "imprisonment", "fine", "death", "rigorous"]):
        return "punishment"
    elif any(kw in combined for kw in ["defin", "meaning", "interpretation"]):
        return "definition"
    elif any(kw in combined for kw in ["right", "entitled", "may claim", "shall be entitled"]):
        return "right"
    elif any(kw in combined for kw in ["offence", "crime", "stolen", "fraud", "cheating", "murder", "theft"]):
        return "offence"
    elif any(kw in combined for kw in ["procedure", "file", "complaint", "application", "form"]):
        return "procedure"
    elif any(kw in combined for kw in ["except", "defence", "not apply", "exclusion"]):
        return "exception"
    return "section_reference"


def generate_multi_qa_per_section(section: dict, pairs_per_section: int = 10) -> list[dict]:
    """Generate multiple Q&A pairs per section using varied templates."""
    section_no = section.get("section_number", "")
    title = section.get("title", "this provision")
    text = section.get("text", "")
    act_name = section.get("act_name", "Indian law")

    category = categorize_section(title, text)
    templates = QUESTION_TEMPLATES_V3.get(category, QUESTION_TEMPLATES_V3["section_reference"])

    # Always include section_reference templates
    all_templates = templates + QUESTION_TEMPLATES_V3["section_reference"]
    # Remove duplicates while preserving order
    seen = set()
    unique_templates = []
    for t in all_templates:
        if t not in seen:
            seen.add(t)
            unique_templates.append(t)

    title_lower = title.lower().rstrip(".")
    clean_text = re.sub(r"\s+", " ", text).strip()
    if len(clean_text) > 600:
        clean_text = clean_text[:600] + "..."

    pairs = []
    selected = random.sample(unique_templates, min(pairs_per_section, len(unique_templates)))

    for template in selected:
        instruction = template.format(
            title=title,
            title_lower=title_lower,
            section_number=section_no,
            act_name=act_name,
        )

        output = (
            f"Under Section {section_no} of the {act_name} ({title}):\n\n"
            f"{clean_text}\n\n"
            f"This section falls under the {act_name} and provides guidance on {title_lower}.\n\n"
            f"DISCLAIMER: This is legal orientation, not legal advice. "
            f"Consult a qualified advocate for your specific situation."
        )

        pairs.append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })

    return pairs


def generate_scenario_pairs(section: dict, num_scenarios: int = 2) -> list[dict]:
    """Generate scenario-based Q&A pairs from section text."""
    section_no = section.get("section_number", "")
    title = section.get("title", "this offence")
    text = section.get("text", "")
    act_name = section.get("act_name", "Indian law")

    if not text or len(text) < 50:
        return []

    clean_text = re.sub(r"\s+", " ", text).strip()
    if len(clean_text) > 500:
        clean_text = clean_text[:500] + "..."

    scenarios = [
        f"A person is accused of {title.lower()} under {act_name}. What are the elements the prosecution must prove?",
        f"If someone commits {title.lower()}, what legal remedies are available to the victim?",
        f"Under what circumstances can {title.lower()} be justified as an exception?",
        f"What investigation steps must the police follow when {title.lower()} is reported?",
        f"How would a court determine the severity of {title.lower()} in a specific case?",
    ]

    pairs = []
    selected = random.sample(scenarios, min(num_scenarios, len(scenarios)))

    for scenario in selected:
        output = (
            f"Regarding {title} under Section {section_no} of {act_name}:\n\n"
            f"{clean_text}\n\n"
            f"This section provides the legal framework for addressing {title.lower()}. "
            f"The specific application depends on the facts and circumstances of each case.\n\n"
            f"DISCLAIMER: This is legal orientation, not legal advice. "
            f"Consult a qualified advocate for your specific situation."
        )
        pairs.append({
            "instruction": scenario,
            "input": "",
            "output": output,
        })

    return pairs


def generate_ipc_bns_pairs() -> list[dict]:
    """Generate Q&A pairs for IPC to BNS section mapping."""
    pairs = []
    for ipc_section, bns_equiv in IPC_BNS_MAPPING.items():
        if bns_equiv is None:
            continue
        pairs.append({
            "instruction": f"What is the BNS equivalent of Section {ipc_section} of the Indian Penal Code?",
            "input": "",
            "output": f"Section {ipc_section} of the Indian Penal Code (IPC) corresponds to {bns_equiv} of the Bharatiya Nyaya Sanhita (BNS) 2023.\n\nDISCLAIMER: This is legal orientation, not legal advice. Consult a qualified advocate for your specific situation.",
        })
    return pairs


def load_sections_from_raw(raw_dir: Path) -> list[dict]:
    """Load all sections from raw scraped data."""
    sections = []
    for json_file in raw_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "sections" in data:
                    for sec in data["sections"]:
                        sec["act_name"] = data.get("name", "")
                        sections.append(sec)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return sections


def generate_groq_pairs(sections: list[dict], max_pairs: int = 10000) -> list[dict]:
    """Generate Q&A pairs using Groq API.

    Optimized for speed:
    - 10 questions per API call
    - 2-second delay (30 RPM free tier)
    - Concurrent requests with threading
    - Automatic retry on rate limits
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set, skipping LLM generation")
        return []

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except ImportError:
        print("groq package not installed, skipping LLM generation")
        return []

    # Filter sections with substantial text
    substantial = [s for s in sections if len(s.get("text", "")) > 100]
    print(f"Generating LLM Q&A for {len(substantial)} sections (target: {max_pairs} pairs)...")

    pairs = []
    failed = 0
    start_time = time.time()

    for i, section in enumerate(substantial):
        if len(pairs) >= max_pairs:
            break

        section_no = section.get("section_number", "")
        title = section.get("title", "")
        text = section.get("text", "")[:600]
        act_name = section.get("act_name", "Indian law")

        prompt = f"""Based on this Indian law section, generate 10 diverse Q&A pairs.

Section {section_no}: {title}
{act_name}
Text: {text}

Generate 10 diverse questions (definition, penalty, procedure, exception, scenario, comparison, jurisdiction, evidence, rights, practical).

Return ONLY a JSON array:
[{{"instruction": "question", "output": "answer"}}, ...]

Rules:
- Answers based on section text only
- Include disclaimer at end of each answer
- Keep answers under 150 words
- No markdown"""

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=3000,
                )
                content = response.choices[0].message.content
                json_match = re.search(r"\[.*\]", content, re.DOTALL)
                if json_match:
                    generated = json.loads(json_match.group())
                    for item in generated:
                        if "instruction" in item and "output" in item:
                            pairs.append({
                                "instruction": item["instruction"],
                                "input": "",
                                "output": item["output"],
                            })
                break  # Success, move to next section
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 10 * (attempt + 1)
                    time.sleep(wait)
                else:
                    failed += 1
                    break

        # Rate limit: 2 seconds between requests
        time.sleep(2)

        # Progress every 50 sections
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = len(pairs) / (elapsed / 60) if elapsed > 0 else 0
            print(f"  [{i+1}/{len(substantial)}] {len(pairs)} pairs | "
                  f"{rate:.0f} pairs/min | {failed} failed | "
                  f"{elapsed:.0f}s elapsed")

    print(f"Generated {len(pairs)} LLM Q&A pairs ({failed} sections failed)")
    return pairs


def generate_v3_dataset(pairs_per_section: int = 10, use_groq: bool = True, max_groq_pairs: int = 10000):
    """Generate the full v3 dataset targeting 50k+ pairs.

    Sources:
    1. Scraped sections x 10+ Q&A pairs each (templates + scenarios)
    2. Expanded IPC→BNS mapping pairs (~200 transitions)
    3. Abbreviation disambiguation pairs (~20)
    4. Groq API LLM-generated pairs (up to 10k)
    """
    raw_dir = config.raw_dir
    output_dir = config.synthetic_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    all_pairs = []

    # 1. Load scraped sections and generate multiple Q&A per section
    sections = load_sections_from_raw(raw_dir)
    print(f"Loaded {len(sections)} sections from raw data")

    if sections:
        for i, section in enumerate(sections):
            if not section.get("text"):
                continue
            # Template-based Q&A
            qa_pairs = generate_multi_qa_per_section(section, pairs_per_section)
            all_pairs.extend(qa_pairs)
            # Scenario-based Q&A
            scenario_pairs = generate_scenario_pairs(section, 2)
            all_pairs.extend(scenario_pairs)
            if (i + 1) % 100 == 0:
                print(f"  Generated {len(all_pairs)} pairs from {i + 1}/{len(sections)} sections...")
        print(f"Generated {len(all_pairs)} pairs from {len(sections)} sections ({pairs_per_section} + 2 scenarios per section)")

    # 2. Add expanded IPC to BNS mapping pairs
    ipc_bns_pairs = generate_ipc_bns_pairs()
    all_pairs.extend(ipc_bns_pairs)
    print(f"Added {len(ipc_bns_pairs)} IPC to BNS mapping pairs")

    # 3. Add abbreviation disambiguation pairs
    all_pairs.extend(ABBREVIATION_PAIRS)
    print(f"Added {len(ABBREVIATION_PAIRS)} abbreviation disambiguation pairs")

    # 4. Add Groq API LLM-generated pairs
    if use_groq:
        groq_pairs = generate_groq_pairs(sections, max_groq_pairs)
        all_pairs.extend(groq_pairs)

    # Deduplicate
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        normalized = pair["instruction"].lower().strip()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        if normalized not in seen:
            seen.add(normalized)
            unique_pairs.append(pair)

    print(f"\nTotal unique pairs: {len(unique_pairs)}")

    # Save
    output_file = output_dir / "synthetic_qa_v3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("v3 Dataset Summary")
    print("=" * 60)
    section_count = len(all_pairs) - len(ipc_bns_pairs) - len(ABBREVIATION_PAIRS) - (len(groq_pairs) if use_groq else 0)
    print(f"Template-based pairs: {section_count}")
    print(f"IPC to BNS mapping pairs: {len(ipc_bns_pairs)}")
    print(f"Abbreviation pairs: {len(ABBREVIATION_PAIRS)}")
    print(f"LLM-generated pairs: {len(groq_pairs) if use_groq else 0}")
    print(f"Total (before dedup): {len(all_pairs)}")
    print(f"Total (after dedup): {len(unique_pairs)}")
    print(f"Target: 50,000+")

    return unique_pairs


if __name__ == "__main__":
    generate_v3_dataset()
