import matplotlib.pyplot as plt
import json
img = plt.imread("encoded.png")

print("Loded image: {} x {} x {}\n".format(len(img), len(img[0]), len(img[0][0])))

gridX = gridY = gridZ = 10
numProbes = gridX * gridY * gridZ
probeIdx = 0
coeffIdx = 0
sh = []
validityCoeffs = []
probeCoeffs = []

for row in img:
	for pixel in row:
		
		mantissa = float(int((int(pixel[0] * 255.0) << 8) | int(pixel[1] * 255.0)) - 0x7fff) / float(0x7fff)
		exponent = float(int(pixel[2] * 255.0) - 0x7f)
		value = mantissa * pow(2.0, exponent)

		if coeffIdx < 12:
			sh.append(value)	
			#print("{}: [{},{},{}] -> [{},{}] -> {}".format(probeIdx, pixel[0], pixel[1], pixel[2], mantissa, exponent, value))	
		elif coeffIdx == 27:
			sh.append(value)
			#print("{}: [{},{},{}] -> [{},{}] -> {}".format(probeIdx, pixel[0], pixel[1], pixel[2], mantissa, exponent, value))	
		
		if coeffIdx == 29:
			probeCoeffs.extend([sh[0], sh[1], sh[2], sh[3], sh[6], sh[9], sh[4], sh[7], sh[10], sh[5], sh[8], sh[11]])
			sh.clear()
			probeIdx += 1
			coeffIdx = 0
		else:	
			coeffIdx += 1

		if probeIdx == numProbes:
			break
	if probeIdx == numProbes:
		break

jsonDict = {
	"description": "",
	"sampleNum": 0,
	"size":
	{
		"x": 1.0, "y": 1.0, "z":1.0
	},
	"resolution":
	{
		"x": gridX, "y": gridY, "z": gridZ
	},
	"coefficients": probeCoeffs,
	"dataValidity": validityCoeffs
}

with open('data.json', 'w') as jsonFile:
    json.dump(jsonDict, jsonFile, indent=4)

print("Done! Exported {} of {} probes.".format(probeIdx, numProbes))
