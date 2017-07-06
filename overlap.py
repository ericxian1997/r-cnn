def overlap(x1, y1, w1, h1, x2, y2, w2, h2):
	
	endx = max(x1+w1, x2+w2)
	startx = min(x1, x2)
	width = w1+w2-(endx-startx)

	endy = max(y1+h1, y2+h2)
	starty = min(y1, y2)
	height = h1+h2-(endy-starty)

	if width <= 0 or height <= 0:
		area = 0
		ratio = 0
	else:
		area = width*height
		area1 = w1*h1
		area2 = w2*h2
		ratio = float(area)/(area1+area2-area)

	return ratio