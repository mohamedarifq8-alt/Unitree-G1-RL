# =================================================================
#خلية التهيئة وتثبيت الإصدارات الصارمة (Environment Setup & Strict Dependencies).
# =================================================================
%cd /kaggle/working/
!rm -rf mujoco_playground logs __pycache__

!git clone https://github.com/google-deepmind/mujoco_playground.git
%cd /kaggle/working/mujoco_playground
!pip install -e . 2>&1 | tail -3

# تثبيت JAX 0.11.2 + CUDA plugin بنفس الإصدار
!pip install --force-reinstall \
    jax==0.11.2 \
    jaxlib==0.11.2 \
    jax-cuda12-plugin==0.11.2 \
    jax-cuda12-pjrt==0.11.2 \
    2>&1 | tail -3

print("\n✅ التثبيت اكتمل")
print("⚠️  الآن: Run → Restart Session")

# =================================================================
#خلية كتابة كود الترقيع  لمكتبة Brax (Brax Surgical Patch Script).
# =================================================================
%%writefile /tmp/fix_brax_complete.py
import pathlib, re

f = pathlib.Path('/usr/local/lib/python3.13/dist-packages/brax/training/agents/ppo/train.py')
content = f.read_text()
original = content

# ═══ إصلاح 1: device_put_replicated → device_put ═══
# JAX 0.11.2 حذف device_put_replicated
content = content.replace(
    'jax.device_put_replicated',
    'jax.device_put'
)

# ═══ إصلاح 2: device_put مع GPU واحد ═══
# مع GPU واحد، نحتاج expand_dims + device واحد (وليس قائمة)
# الكود الأصلي:
#   training_state = jax.device_put(
#       training_state, jax.local_devices()[:local_devices_to_use]
#   )
# يُستبدل بـ:
#   training_state = jax.tree_util.tree_map(
#       lambda x: jnp.expand_dims(x, 0), training_state
#   )
#   training_state = jax.device_put(
#       training_state, jax.local_devices()[0]
#   )

old_device_put = re.compile(
    r'training_state\s*=\s*jax\.device_put\(\s*\n?\s*training_state,\s*jax\.local_devices\(\)\[:local_devices_to_use\]\s*\n?\s*\)',
    re.MULTILINE
)

new_device_put = '''training_state = jax.tree_util.tree_map(
      lambda x: jnp.expand_dims(x, 0), training_state
  )
  training_state = jax.device_put(
      training_state, jax.local_devices()[0]
  )'''

content = old_device_put.sub(new_device_put, content)

# ═══ إصلاح 3: التأكد من squeeze(0) في _unpmap ═══
# expand_dims أضاف بعد، وsqueeze يزيله عند الحفظ/التقييم
# تأكد أن squeeze(0) موجود (قد يكون محذوفاً من محاولات سابقة)
content = content.replace(
    'lambda x: x.addressable_shards[0].data, v',
    'lambda x: x.addressable_shards[0].data.squeeze(0), v'
)

# ═══ حفظ الملف ═══
f.write_text(content)

# ═══ التحقق ═══
final = f.read_text()
checks = [
    ('device_put_replicated' not in final,
     'device_put_replicated → device_put'),
    ('expand_dims' in final,
     'expand_dims قبل device_put'),
    ('jax.local_devices()[0]' in final,
     'GPU واحد (ليس قائمة)'),
    ('squeeze(0), v' in final,
     'squeeze(0) في _unpmap'),
]

print('\\n' + '=' * 50)
print('  إصلاحات Brax لـ JAX 0.11.2 + GPU واحد')
print('=' * 50)

all_ok = True
for ok, name in checks:
    status = '✅' if ok else '❌'
    print(f'  {status} {name}')
    if not ok:
        all_ok = False

if all_ok:
    print('\\n✅ جميع الإصلاحات مطبقة بنجاح!')
else:
    print('\\n⚠️  بعض الإصلاحات لم تُطبق — جرب التدريب على أي حال')


# =================================================================
#خلية تنفيذ الترقيع (Executing the Patch).
# =================================================================
!python /tmp/fix_brax_complete.py

# =================================================================
# خلية تهيئة ذاكرة GPU والتحقق الأمني (VRAM Config & Sanity Check).
# =================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
print(f"JAX: {jax.__version__} | GPU: {jax.devices()} | Count: {jax.device_count()}")

assert jax.__version__ == '0.11.2', f"❌ JAX يجب أن يكون 0.11.2! لديك: {jax.__version__}"
assert jax.device_count() == 1, f"❌ يجب GPU واحد! لديك: {jax.device_count()}"
print("✅ كل شيء صحيح!")

# =================================================================
#  خليه اختيارية  تستخدم عند مواصلة التدريب على اوزان من مراحل سابقة لنقل الملفات القديمة والتدريب عليها
# =================================================================
%%bash
# 1. إنشاء مجلد logs
mkdir -p /kaggle/working/mujoco_playground/logs/

# 2. البحث عن مسار مجلد G1_staged الفعلي داخل المدخلات
SOURCE_DIR=$(find /kaggle/input -type d -name "G1_staged" | head -n 1)

if [ -z "$SOURCE_DIR" ]; then
  echo "❌ خطأ: لم يتم العثور على مجلد G1_staged في المدخلات."
else
  echo "✅ تم العثور على الملفات في: $SOURCE_DIR"
  
  # 3. نسخ المجلد بمحتوياته
  cp -r "$SOURCE_DIR" /kaggle/working/mujoco_playground/logs/
  
  echo "✅ تم النسخ بنجاح! محتويات التدريب السابقة:"
  ls -lh /kaggle/working/mujoco_playground/logs/G1_staged/
fi

# =================================================================
# خليه كود بيانات التدريب  
# حسب ماهو مبين في الشرحlearning_rateملاحظة في كل مرحلة عليك نسخ الكود وتكراره لكل مرحلة وتغيير ال
# =================================================================
%%writefile /kaggle/working/mujoco_playground/train_g1_staged.py
import os, sys, glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = '0'

import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/kaggle/working/mujoco_playground')

import jax
from mujoco_playground import registry

# =================================================================
# إعدادات المكافآت — مطابقة للقيم المضبوطة (tuned) مع لمسات mjlab الآمنة
# =================================================================
_orig_get_default_config = registry.get_default_config

def _custom_get_default_config(env_name):
    config = _orig_get_default_config(env_name)
    if env_name == 'G1JoystickFlatTerrain':
        with config.unlocked():
            config.reward_config.scales.termination = -100.0
            config.reward_config.scales.alive = 0.0
            config.reward_config.scales.action_rate = -0.01
            config.reward_config.scales.torques = -0.0001
            config.reward_config.scales.dof_acc = -2.5e-7
    return config

registry.get_default_config = _custom_get_default_config
# =================================================================

stage = 1
for arg in sys.argv:
    if arg.startswith('--stage='):
        stage = int(arg.split('=')[1])
sys.argv = [a for a in sys.argv if not a.startswith('--stage=')]

STAGE_CONFIG = {
    1: {'timesteps': 40_000_000, 'resume': False},
    2: {'timesteps': 40_000_000, 'resume': True},
    3: {'timesteps': 40_000_000, 'resume': True},
    4: {'timesteps': 40_000_000, 'resume': True},
    5: {'timesteps': 40_000_000, 'resume': True},
}

BASE_DIR = '/kaggle/working/mujoco_playground/logs/G1_staged'

def get_latest_checkpoint():
    ckpt_dirs = glob.glob(f'{BASE_DIR}/*/checkpoints')
    valid_dirs = []
    for d in ckpt_dirs:
        if not os.path.isdir(d): continue
        subdirs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s)) and s.isdigit()]
        if subdirs:
            valid_dirs.append((d, max(int(s) for s in subdirs)))
    if not valid_dirs: return None
    valid_dirs.sort(key=lambda x: os.path.getctime(x[0]), reverse=True)
    return valid_dirs[0][0]

config = STAGE_CONFIG.get(stage)

print("=" * 55)
print(f"  🤖 تدريب G1 | المرحلة {stage}/5")

if config['resume']:
    ckpt = get_latest_checkpoint()
    if ckpt: print(f"  🔄 استئناف من أحدث checkpoint: {ckpt}")
    else: 
        print("  ⚠️ لم يتم العثور على أوزان سابقة - بداية من الصفر")
        config['resume'] = False

from absl import app
from learning import train_jax_ppo

sys.argv = [
    'train_jax_ppo',
    '--env_name=G1JoystickFlatTerrain',
    '--num_envs=4096',                          
    '--num_eval_envs=64',
    '--policy_hidden_layer_sizes=512,256,128',  
    '--value_hidden_layer_sizes=512,256,128',   
    '--policy_obs_key=state',
    '--value_obs_key=privileged_state',         
    '--unroll_length=20',                       
    '--batch_size=256',                         
    '--num_minibatches=16',                     
    '--num_updates_per_batch=4',
    '--num_evals=10',
    f'--num_timesteps={config["timesteps"]}',
    '--episode_length=1000',
    '--learning_rate=3e-4',                     
    '--entropy_cost=0.005',                     
    '--discounting=0.97',                       
    '--clipping_epsilon=0.2',                   
    '--reward_scaling=1.0',                     
    '--max_grad_norm=1.0',
    '--num_videos=0',
    f'--logdir={BASE_DIR}'
]

if config['resume'] and ckpt:
    sys.argv.append(f'--load_checkpoint_path={ckpt}')

app.run(train_jax_ppo.main)

# =================================================================
# خلية كود ملف حفظ فيديو التدريب 
# =================================================================
%%writefile /kaggle/working/mujoco_playground/record_g1_staged.py
import os, sys, glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/kaggle/working/mujoco_playground')

from mujoco_playground import registry

# =================================================================
# 🔥 اعتراض دالة التصوير لإجبار الكاميرا على تتبع الروبوت
# =================================================================
original_load = registry.load

def custom_load(*args, **kwargs):
    env = original_load(*args, **kwargs)
    orig_render = env.render
    
    def tracking_render(state, *r_args, **r_kwargs):
        # توجيه المحاكي لاستخدام كاميرا 'track' التي تتبع مركز الروبوت
        r_kwargs['camera'] = 'track' 
        try:
            return orig_render(state, *r_args, **r_kwargs)
        except Exception:
            r_kwargs.pop('camera', None)
            return orig_render(state, *r_args, **r_kwargs)
            
    env.render = tracking_render
    return env

registry.load = custom_load
# =================================================================

ckpt_dirs = glob.glob('/kaggle/working/mujoco_playground/logs/G1_staged/*/checkpoints')
valid_dirs = [d for d in ckpt_dirs if os.path.isdir(d) and any(s.isdigit() for s in os.listdir(d))]

if not valid_dirs:
    print("❌ لم يتم العثور على نماذج محفوظة!")
    sys.exit(1)

latest_dir = max(valid_dirs, key=lambda x: os.path.getctime(x))
print(f"🎬 جاري تسجيل فيديو لأحدث تدريب...")

from absl import app
from learning import train_jax_ppo

if __name__ == '__main__':
    # تم إزالة config_overrides تماماً من هنا لتجنب خطأ FATAL Flags
    sys.argv = [
        'train_jax_ppo',
        '--env_name=G1JoystickFlatTerrain',
        '--play_only',
        f'--load_checkpoint_path={latest_dir}',
        '--num_videos=1',
        '--episode_length=1000',
        '--num_envs=1',
        '--num_eval_envs=1',
        '--policy_hidden_layer_sizes=512,256,128', 
        '--value_hidden_layer_sizes=512,256,128',
    ]
    app.run(train_jax_ppo.main)
    

# =================================================================
# خلية تشغيل التدريب للمراحل , ملاحظة عندما  تريد تشغيل مرحلة جديدة غير الرقم (1) في الكود التالي الى رقم المرحلة الجديدة
# =================================================================
!CUDA_VISIBLE_DEVICES=0 python /kaggle/working/mujoco_playground/train_g1_staged.py --stage=1

#خليه تشغيل عملية الحفظ للفيديو
!CUDA_VISIBLE_DEVICES=0 python /kaggle/working/mujoco_playground/record_g1_staged.py

#خليه حفظ التدريب وملف الاوزان المضغوط
%%bash
cd /kaggle/working/
tar -czf G1_Stage1_Perfect_backup.tar.gz -C /kaggle/working/mujoco_playground/logs/ G1_staged/