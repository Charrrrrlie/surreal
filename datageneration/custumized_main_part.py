import sys
import os
import random

import bpy
import numpy as np
from os import getenv
from os import remove
from os.path import join, dirname, realpath, exists
from mathutils import Matrix, Vector, Quaternion, Euler
from glob import glob
from random import choice

from pickle import load

sys.path.append(os.getcwd())
from custumized_utils import init_envs, allocate_output_dict, apply_trans_pose_shape, reset_joint_positions, get_bone_locs

sys.path.insert(0, ".")

# load poses and shapes
def load_body_data(smpl_data, ob, obname, gender='female', idx=0):
    # load MoSHed data from CMU Mocap (only the given idx is loaded)
    
    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))
    
    name = sorted(cmu_keys)[idx % len(cmu_keys)]
    
    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {'poses':smpl_data[seq],
                                                   'trans':smpl_data[seq.replace('pose_','trans_')]}

    # compute the number of shape blendshapes in the model
    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return(cmu_parms, fshapes, name)

def load_dataset(params, ob, obname, gender):
    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    use_split = params['use_split']

    ishape = params['ishape']
    idx = params['idx']
    stepsize = params['stepsize']
    clipsize = params['clipsize']
    stride = params['stride']
    
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))
    cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname, idx=idx, gender=gender)

    nb_fshapes = len(fshapes)
    if use_split == 'train':
        fshapes = fshapes[:int(nb_fshapes*0.8)]
    elif use_split == 'test':
        fshapes = fshapes[int(nb_fshapes*0.8):]

    shape = choice(fshapes) #+random_shape(.5) can add noise

    data = cmu_parms[name]

    fbegin = ishape*stepsize*stride
    fend = min(ishape*stepsize*stride + stepsize*clipsize, len(data['poses']))

    # force recomputation of joint angles unless shape is all zeros
    curr_shape = np.zeros_like(shape)
    nframes = len(data['poses'][::stepsize])

    return data['poses'][fbegin:fend:stepsize], shape, data['trans'][fbegin:fend:stepsize],\
        smpl_data['regression_verts'], smpl_data['joint_regressor']


import time
start_time = None
def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))

def main():
    # time logging
    global start_time
    start_time = time.time()

    import argparse
    
    # parse commandline arguments
    log_message(sys.argv)
    parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    parser.add_argument('--ishape', type=int,
                        help='requested cut, according to the stride')
    parser.add_argument('--stride', type=int,
                        help='stride amount, default 50')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    idx = args.idx
    ishape = args.ishape
    stride = args.stride
    
    log_message("input idx: %d" % idx)
    log_message("input ishape: %d" % ishape)
    log_message("input stride: %d" % stride)
    
    if idx == None:
        exit(1)
    if ishape == None:
        exit(1)
    if stride == None:
        log_message("WARNING: stride not specified, using default value 50")
        stride = 50
    
    log_message("output idx: %d" % idx)

    # import configuration
    log_message("Importing configuration")
    import config
    params = config.load_file('config', 'SYNTH_DATA')
    params['idx'] = idx
    params['ishape'] = ishape
    params['stride'] = stride
    
    params['genders'] = {0: 'female', 1: 'male'}

    
    ############## utils here ######
    scene, scs, ob, obname, arm_ob, cam_ob, gender, params = init_envs(params)
    ################

    
    # TODO(yyc): load data
    data_pose, data_shape, data_trans, regresson_verts, joint_regressor = load_dataset(params, ob, obname, gender)
    dict_info = allocate_output_dict(len(data_pose), params)
    
    orig_pelvis_loc = (arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()) - Vector((-1., 1., 1.))
    orig_cam_loc = cam_ob.location.copy()

    orig_trans = np.asarray(arm_ob.pose.bones[obname+'_Pelvis'].location).copy()

    # for each clipsize'th frame in the sequence
    get_real_frame = lambda ifr: ifr
    random_zrot = 0
    reset_loc = False
    batch_it = 0
    curr_shape = reset_joint_positions(orig_trans, data_shape, ob, arm_ob, obname, scene,
                                       cam_ob, regresson_verts, joint_regressor)
    random_zrot = 2*np.pi*np.random.rand()
    
    arm_ob.animation_data_clear()
    cam_ob.animation_data_clear()

    # create a keyframe animation with pose, translation, blendshapes and camera motion
    # LOOP TO CREATE 3D ANIMATION
    for seq_frame, (pose, trans) in enumerate(zip(data_pose, data_trans)):
        iframe = seq_frame
        scene.frame_set(get_real_frame(seq_frame))

        # apply the translation, pose and shape to the character
        apply_trans_pose_shape(Vector(trans), pose, data_shape, ob, arm_ob, obname, scene, cam_ob, get_real_frame(seq_frame))
        dict_info['shape'][:, iframe] = data_shape
        dict_info['pose'][:, iframe] = pose
        dict_info['gender'][iframe] = list(params['genders'])[list(params['genders'].values()).index(gender)]
        if(params['output_types']['vblur']):
            dict_info['vblur_factor'][iframe] = params['vblur_factor']

        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot), 'XYZ'))
        arm_ob.pose.bones[obname+'_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
        dict_info['zrot'][iframe] = random_zrot

        scene.update()

        # Bodies centered only in each minibatch of clipsize frames
        if seq_frame == 0 or reset_loc: 
            reset_loc = False
            new_pelvis_loc = arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname+'_Pelvis'].head.copy()
            cam_ob.location = orig_cam_loc.copy() + (new_pelvis_loc.copy() - orig_pelvis_loc.copy())
            cam_ob.keyframe_insert('location', frame=get_real_frame(seq_frame))
            dict_info['camLoc'] = np.array(cam_ob.location)

    # random light
    sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
    sh_coeffs[0] = .5 + .9 * np.random.rand() # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
    sh_coeffs[1] = -.7 * np.random.rand()

    for ish, coeff in enumerate(sh_coeffs):
        for sc in scs:
            sc.inputs[ish+1].default_value = coeff

    # iterate over the keyframes and render
    # LOOP TO RENDER
    for seq_frame, (pose, trans) in enumerate(zip(data_pose, data_trans)):
        scene.frame_set(get_real_frame(seq_frame))
        iframe = seq_frame

        # dict_info['bg'][iframe] = bg_img_name
        # dict_info['cloth'][iframe] = cloth_img_name
        # dict_info['light'][:, iframe] = sh_coeffs

        scene.render.use_antialiasing = False
        scene.render.filepath = join(params['rgb_path'], 'Image%04d.png' % get_real_frame(seq_frame))

        log_message("Rendering frame %d" % seq_frame)
        
        # disable render output
        logfile = '/dev/null'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        # Render
        bpy.ops.render.render(write_still=True)

        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)

        # bone locations should be saved after rendering so that the bones are updated
        bone_locs_2D, bone_locs_3D = get_bone_locs(obname, arm_ob, scene, cam_ob)
        dict_info['joints2D'][:, :, iframe] = np.transpose(bone_locs_2D)
        dict_info['joints3D'][:, :, iframe] = np.transpose(bone_locs_3D)

        reset_loc = (bone_locs_2D.max(axis=-1) > 256).any() or (bone_locs_2D.min(axis=0) < 0).any()
        arm_ob.pose.bones[obname+'_root'].rotation_quaternion = Quaternion((1, 0, 0, 0))

    final_output_path = join(params['output_path'], 'run{}_{}'.format(idx, ishape))
    if not exists(final_output_path):
        os.makedirs(final_output_path)

    # save RGB data with ffmpeg (if you don't have h264 codec, you can replace with another one and control the quality with something like -q:v 3)
    cmd_ffmpeg = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 ''%s/c%04d.mp4''' % (join(params['rgb_path'], 'Image%04d.png'), final_output_path, (ishape))
    log_message("Generating RGB video (%s)" % cmd_ffmpeg)
    os.system(cmd_ffmpeg)

    # if(params['output_types']['vblur']):
    #     cmd_ffmpeg_vblur = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ''%s/c%04d.mp4''' % (join(params['res_paths']['vblur'], 'Image%04d.png'), params['output_path']+'_vblur', (ishape + 1))
    #     log_message("Generating vblur video (%s)" % cmd_ffmpeg_vblur)
    #     os.system(cmd_ffmpeg_vblur)

    # if(params['output_types']['fg']):
    #     cmd_ffmpeg_fg = 'ffmpeg -y -r 30 -i ''%s'' -c:v h264 -pix_fmt yuv420p -crf 23 ''%s/c%04d.mp4''' % (join(params['res_paths'] ['fg'], 'Image%04d.png'), params['output_path']+'_fg', (ishape + 1))
    #     log_message("Generating fg video (%s)" % cmd_ffmpeg_fg)
    #     os.system(cmd_ffmpeg_fg)
   
    # cmd_tar = 'tar -czvf %s/%s.tar.gz -C %s %s' % (output_path, rgb_dirname, tmp_path, rgb_dirname)
    # log_message("Tarballing the images (%s)" % cmd_tar)
    # os.system(cmd_tar)
    
    # save annotation excluding png/exr data to _info.mat file
    import scipy.io
    scipy.io.savemat(join(final_output_path, 'c%04d_info.mat' % ishape), dict_info, do_compression=True)

if __name__ == '__main__':
    main()
