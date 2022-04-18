import os
import synthetic_artifects_utils
import argparse

def main():
    parser = argparse.ArgumentParser(description='!!!Make Object-level Benchmark Dataset!!!')
    # Base Dir setting
    parser.add_argument('--Base_dir_in', '-bdi', type=str, default=None, help='path of the scan pcs input')
    parser.add_argument('--Base_dir_Out', '-bdo', type=str, default='Out', help='path of the scan pcs output')
    parser.add_argument('--NonUniform_', '-nonu', type=str, default='Nonuniform_Normal', help='Nonuniform psc input/output children floder')
    parser.add_argument('--Noise_in', '-noi', type=str, default='Noise_Normal', help='Noise psc input children floder')

    parser.add_argument('--NonUniform_out', '-nono', type=str, default='Nonuniform', help='Nonuniform psc output children floder')
    parser.add_argument('--Noise_out', '-noo', type=str, default='Noise', help='Noise psc output children floder')
    parser.add_argument('--Uniform_', '-un', type=str, default='Uniform', help='Noise psc input children floder')
    parser.add_argument('--Outlier_', '-out', type=str, default='Outlier', help='Outlier psc input children floder')
    parser.add_argument('--MissingData_', '-md', type=str, default='Missing_Data', help='Missing data psc input children floder')
    parser.add_argument('--Misalignment_', '-ma', type=str, default='Misalignment', help='Missalignment psc input children floder')

    # Hyper parameters
    parser.add_argument('--sever_level', '-sl', type=int, metavar=[1,2,3], default=1, help='Artifacts level')
    parser.add_argument('--which_artifact', '-w', type=str, metavar=['Uniform', 'Nonuniform', 'Noise', 'Outlier', 'MissingData', 'Misalignment'], default='Uniform', help='Artifacts level')
    parser.add_argument("--devices", "-d", type=str, default=["0"], help="gpu to be used")
    parser.add_argument('--anoise', '-an', type=float, metavar=[0.001,0.003,0.006], default=0.001)
    parser.add_argument('--outlier_number', '-on', type=float, default=0.001)
    parser.add_argument('--outlier_intensity', '-oi', type=float, default=0.1)
    parser.add_argument('--Misalignment_intensity', '-mi', type=float,  default=1/20.)
    parser.add_argument('--Misalignment_angle', '-mma', type=float, default=1.)
    parser.add_argument('--num_worker', '-nw', type=int, default=1)

    args = parser.parse_args()
    print(args)

    Base_dir_in = os.path.abspath(args.Base_dir_in)
    NonUniform_ = args.NonUniform_             
    Noise_ = args.Noise_in

    Base_dir_Out = os.path.join(Base_dir_in, args.Base_dir_Out)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.devices)

    num_work = int(args.num_worker)

    ########################################################################
    # Get Uniform Data  fps
    ########################################################################
    if args.which_artifact == 'Uniform':
        In_Dir = os.path.join(Base_dir_in, NonUniform_)
        Out_Dir = os.path.join(Base_dir_Out, args.Uniform_)
        print(In_Dir, Out_Dir)
        synthetic_artifects_utils.Get_Uniform(In_Dir, Out_Dir, FPS=True, num_work=num_work)
    ########################################################################
    # Get Nonuniform Data   random sample
    ########################################################################
    elif args.which_artifact == 'Nonuniform':
        In_Dir = os.path.join(Base_dir_in, NonUniform_)
        Out_Dir = os.path.join(Base_dir_Out, args.NonUniform_out)
        print(In_Dir, Out_Dir)
        synthetic_artifects_utils.Get_Noniform(In_Dir, Out_Dir, num_work=num_work)
    ########################################################################
    # Get Noise Data   fps
    ########################################################################
    elif args.which_artifact == 'Noise':
        In_DirN = os.path.join(Base_dir_in, Noise_)
        In_DirV = os.path.join(Base_dir_in, NonUniform_)
        Out_Dir = os.path.join(Base_dir_Out, args.Noise_out+str(args.sever_level))
        print(In_DirN, In_DirV, Out_Dir, args.anoise)
        # anoise=0.001  0.003  0.006
        synthetic_artifects_utils.Get_Noise(In_DirN, In_DirV, Out_Dir, anoise = args.anoise, FPS=True, num_work=num_work)
    ########################################################################
    # Get Outlier Data
    ########################################################################
    elif args.which_artifact == 'Outlier':
        In_Dir = os.path.join(Base_dir_in, NonUniform_)
        Out_Dir = os.path.join(Base_dir_Out, args.Outlier_+str(args.sever_level))
        print(In_Dir, Out_Dir, args.outlier_number, args.outlier_intensity)
        # number 0.01, 0.15, 0.25  intensity 0.1
        synthetic_artifects_utils.Get_outlier(In_Dir, Out_Dir, number = args.outlier_number, intensity=args.outlier_intensity, FPS=True, num_work=num_work)
    ########################################################################
    # Get Missing Data   fps
    ########################################################################
    elif args.which_artifact == 'MissingData':
        In_Dir = os.path.join(Base_dir_in, NonUniform_)
        Out_Dir = os.path.join(Base_dir_Out, args.MissingData_ +str(args.sever_level))
        print(In_Dir, Out_Dir, args.sever_level)
        # -3 [20 40 60] + 3
        if args.sever_level == 1:
            synthetic_artifects_utils.Get_MissingData(In_Dir, Out_Dir, zanglelist=[[17,23],[37,43],[57,63]], FPS=True, num_work=num_work)
        elif args.sever_level == 2:
            synthetic_artifects_utils.Get_MissingData(In_Dir, Out_Dir, zanglelist=[[17,23],[37,43]], FPS=True, num_work=num_work)
        elif args.sever_level == 3:
            synthetic_artifects_utils.Get_MissingData(In_Dir, Out_Dir, zanglelist=[[17,23]], FPS=True, num_work=num_work)
        else:
            print('error')
    ########################################################################
    # Get Misalignment Data   fps
    ########################################################################
    elif args.which_artifact == 'Misalignment':
        In_Dir = os.path.join(Base_dir_in, NonUniform_)
        Out_Dir = os.path.join(Base_dir_Out, args.Misalignment_ + str(args.sever_level))
        print(In_Dir, Out_Dir, args.Misalignment_intensity, args.Misalignment_angle)
        # intensity = 0.005  0.01  0.02  angle 0.5 1 2
        synthetic_artifects_utils.Get_Missalignment(In_Dir, Out_Dir, intensity = args.Misalignment_intensity, angle= args.Misalignment_angle, FPS=True, num_work=num_work)
    else:
        print('Error')
    
if __name__ == "__main__":
    main()