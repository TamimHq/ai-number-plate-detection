def overlap(vehicle,plate):

    vx1,vy1,vx2,vy2 = vehicle
    px1,py1,px2,py2 = plate

    ox = max(
        0,
        min(vx2,px2)-max(vx1,px1)
    )

    oy = max(
        0,
        min(vy2,py2)-max(vy1,py1)
    )

    return ox * oy

def assign_plate(vehicle_box,plates):

    vx1,vy1,vx2,vy2 = vehicle_box

    best = None
    best_area = 0

    for p in plates:

        px1,py1,px2,py2 = p["coords"]

        area = overlap(
            vehicle_box,
            p["coords"]
        )

        if area > best_area:

            best_area = area
            best = p

    return best