import sys

pbh_ra = float(sys.argv[1])
pbh_dec = float(sys.argv[2])
loc = str(sys.argv[3])


pbh_string = []
pbh_string.append('<source ROI_Center_Distance="0.0" name="PBH_Source" type="PointSource">\n')
pbh_string.append('	<spectrum apply_edisp="false" type="LogParabola">\n')
pbh_string.append('		<parameter free="1" max="100000" min="1e-07" name="norm" scale="1e-12" value="0.7444094111" />\n')
pbh_string.append('		<parameter free="1" max="3.0" min="1.2" name="alpha" scale="1" value="1.405417681" />\n')
pbh_string.append('		<parameter free="1" max="1.0" min="0.01" name="beta" scale="1" value="0.313444948" />\n')
pbh_string.append('		<parameter free="0" max="500000" min="30" name="Eb" scale="1" value="1000" />\n')
pbh_string.append('	</spectrum>\n')
pbh_string.append('	<spatialModel type="SkyDirFunction">\n')
pbh_string.append('		<parameter free="0" max="360.0" min="-360.0" name="RA" scale="1.0" value="'+str(pbh_ra)+'"/>\n')
pbh_string.append('		<parameter free="0" max="90" min="-90" name="DEC" scale="1.0" value="'+str(pbh_dec)+'"/>\n')
pbh_string.append('	</spatialModel>\n')
pbh_string.append('</source>\n')

file = open(loc+'/xmlmodel.xml','r')
stored_string = []
for line in file:
    stored_string.append(line)
file.close()

file = open(loc+'/xmlmodel.xml','w')
for i in range(len(stored_string)):
    if i == 5:
        for j in range(len(pbh_string)):
            file.write(pbh_string[j])
    file.write(stored_string[i])
file.close()
