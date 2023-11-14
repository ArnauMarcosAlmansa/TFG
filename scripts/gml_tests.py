import glm


def calculate_up_vector(eye, center, world_up):
    forward = glm.normalize(center - eye)
    right = glm.cross(world_up, forward)
    up = glm.cross(forward, right)
    return glm.normalize(up)


pose = glm.mat4x4(
    1, 0, 0, 5,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
)
eye = glm.vec3(5, 0, 0)
center = glm.vec3(0, 0, 0)
up = glm.vec3(0, 1, 0)

cameraUp = calculate_up_vector(eye, center, up)

pose = glm.lookAt(eye, center, cameraUp)

print(pose)
