from iraqitext import IraqiTranslator

s = IraqiTranslator()

text=input("code : ")
print("-"*70)

print(s.to_fusha(text),"to Fusha")
print("-"*70)
print(s.to_iraqi(text),"to Iraqi")
