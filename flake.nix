{
  description = "thox-qk-rope kernel";

  inputs = {
    builder.url = "github:huggingface/kernels";
  };

  outputs =
    { self, builder }:
    builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
      pythonCheckInputs = pkgs: with pkgs; [ numpy ];
    };
}
