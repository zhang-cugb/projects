from subprocess import run


def make_gif(path, name, speed=1, width=400):
    '''
    Function to convert a set of images into a gif.
    Must have ffmpeg in the PATH. Only tested in Windows 7.

    Arguments:
        path(string): path of the images folder
        name(string): base name of images. They must be named as "name%d".
                Example: picture01, picture02,picture03,....
        speed(float) : speed of the gif. The higher the slower.
        width(int) : width in px of the output gif.
    '''

    # Convert png images to video
    a = "ffmpeg -i {0}\{1}%d.png -qscale 0 {0}\{1}.mp4 -y".format(path, name)
    # Create palette
    b = "ffmpeg -i {0}\{1}.mp4 -vf palettegen {0}\{1}_palette.png -y".format(
        path, name)
    # Convert mp4 to gif
    c = "ffmpeg -i {0}\{1}.mp4 -i {0}\{1}_palette.png -filter_complex \
    \"setpts={2}*PTS, scale = {3}:-1:flags=lanczos[x];[x][1:v]paletteuse\" \
    {0}\{1}.gif -y".format(path, name, speed, width)

    # Delete intermediate video
    d = "DEL /Q {}\{}.mp4".format(path, name)
    # Delete palette
    e = "DEL /Q {}\{}_palette.png".format(path, name)

    run(a, shell=True)
    run(b, shell=True)
    run(c, shell=True)
    run(d, shell=True)
    run(e, shell=True)

    return


def delete_files(path, name, ext):
    '''
Function to delete all files that match
path//name*.{ext}
Only tested in Windows 7.

Arguments:
    path, name, ext (string)
'''

    command = "DEL /Q {}\{}*.{}".format(path, name, ext)
    run(command, shell=True)

    return
