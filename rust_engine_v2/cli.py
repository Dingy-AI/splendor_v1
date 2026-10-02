"""Shared opt-in inference switches; default remains fp32/eager."""
from splendor_v1.rust_engine_v2.inference import InferenceOptions


def add_inference_arguments(parser):
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--compile", dest="compile_mode",
                        choices=("off", "default", "reduce-overhead"), default="off")
    parser.add_argument("--profile", action="store_true", help="Include CUDA event timings (adds measurement overhead)")
    parser.add_argument("--bucket-shapes", action="store_true", help="Pad batch/action shapes into reusable buckets")
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)


def inference_options(args):
    return InferenceOptions(precision=args.precision, compile_mode=args.compile_mode,
                            profile=args.profile, bucket_shapes=args.bucket_shapes)
