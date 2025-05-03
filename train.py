import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import soundfile as sf
import json
import pandas as pd
import joblib

def load_audio_data(directory):
    X = []
    y = []
    class_labels = {}
    diagnoses = {}
    file_paths = []
    
    # Load metadata
    metadata_path = os.path.join(directory, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✅ تم تحميل البيانات الوصفية بنجاح")
            print(f"🔍 عدد التصنيفات في البيانات الوصفية: {len(metadata)}")
        except Exception as e:
            print(f"❌ خطأ في تحميل البيانات الوصفية: {e}")
            metadata = {}
    else:
        print(f"❌ ملف البيانات الوصفية غير موجود في {metadata_path}")
        metadata = {}
    
    # Process each audio file
    for audio_file in os.listdir(directory):
        if not audio_file.endswith(('.wav', '.mp3')):
            continue
            
        audio_path = os.path.join(directory, audio_file)
        print(f"⏳ جاري معالجة الملف: {audio_file}")
        
        # Get diagnosis from metadata
        file_info = metadata.get(audio_file, {})
        diagnosis = file_info.get('diagnosis', 'Unknown')
        
        if diagnosis not in class_labels.values():
            class_id = len(class_labels)
            class_labels[class_id] = diagnosis
            diagnoses[class_id] = diagnosis
            print(f"➕ تمت إضافة تصنيف جديد: {diagnosis}")
        
        try:
            audio_data, sample_rate = sf.read(audio_path)
            
            # Preprocess audio
            features = preprocess_audio(audio_data, sample_rate)
            X.append(features)
            y.append(list(class_labels.keys())[list(class_labels.values()).index(diagnosis)])
            file_paths.append(audio_path)
            
        except Exception as e:
            print(f"❌ خطأ في معالجة {audio_file}: {str(e)}")
            continue
    
    if len(X) == 0:
        print("❌ لم يتم العثور على ملفات صوتية صالحة في المجلد المحدد.")
        return None, None, None, None, None
    
    print(f"✅ تمت معالجة {len(X)} ملف صوتي بنجاح")
    return np.array(X), np.array(y), class_labels, diagnoses, file_paths

def preprocess_audio(audio_data, sample_rate=22050):
    try:
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sample_rate != 22050:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        return mfccs_scaled
    except Exception as e:
        print(f"❌ خطأ في معالجة الصوت: {str(e)}")
        raise

def create_model():
    try:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        return model
    except Exception as e:
        print(f"❌ خطأ في إنشاء النموذج: {str(e)}")
        raise

def save_training_report(model, X_train_scaled, X_val_scaled, y_train, y_val, class_labels):
    # Generate training report
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    report = {
        "training_accuracy": model.score(X_train_scaled, y_train),
        "validation_accuracy": model.score(X_val_scaled, y_val),
        "training_report": classification_report(y_train, train_pred, target_names=[class_labels[i] for i in range(len(class_labels))], output_dict=True),
        "validation_report": classification_report(y_val, val_pred, target_names=[class_labels[i] for i in range(len(class_labels))], output_dict=True)
    }
    
    with open('training_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    return report

def main():
    try:
        print("\n🚀 بدء عملية التدريب...")
        
        # Load and preprocess data
        data_dir = "saved_data"
        X, y, class_labels, diagnoses, file_paths = load_audio_data(data_dir)
        
        if X is None:
            return
        
        print(f"\n📊 إحصائيات البيانات:")
        print(f"   - عدد العينات الصوتية: {len(X)}")
        print(f"   - عدد التصنيفات: {len(class_labels)}")
        print("\n📝 التصنيفات المتاحة:")
        for class_id, diagnosis in diagnoses.items():
            count = np.sum(y == class_id)
            print(f"   - {diagnosis}: {count} عينة")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        print("\n⚖️ تطبيع البيانات...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create and train model
        print("🧠 إنشاء وتدريب النموذج...")
        model = create_model()
        model.fit(X_train_scaled, y_train)
        
        # Generate and save training report
        print("\n📈 توليد تقرير التدريب...")
        report = save_training_report(model, X_train_scaled, X_val_scaled, y_train, y_val, class_labels)
        
        # Save model and related files
        print("\n💾 حفظ النموذج والملفات المرتبطة...")
        joblib.dump(model, 'model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        np.save('class_labels.npy', class_labels)
        np.save('diagnoses.npy', diagnoses)
        
        print("\n✅ اكتمل التدريب بنجاح!")
        print(f"📊 دقة التدريب: {report['training_accuracy']:.2%}")
        print(f"📊 دقة التحقق: {report['validation_accuracy']:.2%}")
        
        print("\n📁 تم حفظ الملفات التالية:")
        print("   - model.joblib (النموذج المدرب)")
        print("   - scaler.joblib (معايير التطبيع)")
        print("   - class_labels.npy (تصنيفات الأصوات)")
        print("   - diagnoses.npy (التشخيصات)")
        print("   - training_report.json (تقرير التدريب)")
        
    except Exception as e:
        print(f"\n❌ حدث خطأ أثناء التدريب: {str(e)}")
        raise

if __name__ == "__main__":
    main()