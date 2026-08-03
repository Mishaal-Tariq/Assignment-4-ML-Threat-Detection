#!/usr/bin/env python3
"""
THE ARZENS Synthetic Network Data Generator
Generates realistic network intrusion detection dataset
Compatible with CICIDS2017/UNSW-NB15 format
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import argparse
import os

class NetworkDataGenerator:
    """
    Generates synthetic network traffic data for ML training
    Includes both benign and attack traffic patterns
    """
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Network parameters
        self.protocols = ['TCP', 'UDP', 'ICMP']
        self.services = ['HTTP', 'HTTPS', 'FTP', 'SSH', 'DNS', 'SMTP', 'POP3', 'NNTP', 'IRC', 'OTHER']
        # FIXED: 11 flags with 11 probabilities
        self.flags = ['SF', 'S0', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'SH', 'S1', 'S2', 'S3', 'OTH']
        
        # IP ranges
        self.private_ips = [f"192.168.1.{i}" for i in range(1, 255)] + \
                          [f"10.0.0.{i}" for i in range(1, 255)] + \
                          [f"172.16.0.{i}" for i in range(1, 255)]
        self.public_ips = [f"203.0.113.{i}" for i in range(1, 255)] + \
                          [f"198.51.100.{i}" for i in range(1, 255)] + \
                          [f"192.0.2.{i}" for i in range(1, 255)]
        
        # Attack types
        self.attack_types = {
            'Benign': 0,
            'DoS': 1,
            'PortScan': 2,
            'BruteForce': 3,
            'WebAttack': 4,
            'Botnet': 5,
            'Infiltration': 6
        }
    
    def generate_timestamp(self, start_date=None, days=1):
        """Generate random timestamps over specified days"""
        if start_date is None:
            start_date = datetime(2026, 7, 20, 8, 0, 0)
        
        random_seconds = np.random.randint(0, days * 24 * 3600)
        return start_date + timedelta(seconds=int(random_seconds))
    
    def generate_benign_traffic(self):
        """Generate normal network traffic features"""
        duration = np.random.exponential(50)  # Average 50 seconds
        duration = min(duration, 3600)  # Cap at 1 hour
        
        protocol = np.random.choice(self.protocols, p=[0.7, 0.25, 0.05])
        service = np.random.choice(self.services, p=[0.4, 0.3, 0.05, 0.05, 0.1, 0.03, 0.02, 0.02, 0.01, 0.02])
        # FIXED: 11 probabilities for 11 flags
        flag = np.random.choice(self.flags, p=[0.5, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02])
        
        src_ip = random.choice(self.private_ips)
        dst_ip = random.choice(self.public_ips)
        
        # Normal traffic patterns
        src_port = random.randint(1024, 65535)
        dst_port = random.choice([80, 443, 22, 53, 21, 25, 110, 119])
        
        # Packet counts (normal distribution)
        fwd_packets = max(1, int(np.random.normal(10, 5)))
        bwd_packets = max(1, int(np.random.normal(8, 4)))
        total_packets = fwd_packets + bwd_packets
        
        # Byte counts
        fwd_bytes = fwd_packets * random.randint(40, 1500)
        bwd_bytes = bwd_packets * random.randint(40, 1500)
        total_bytes = fwd_bytes + bwd_bytes
        
        # Flow features
        flow_bytes_s = total_bytes / duration if duration > 0 else 0
        flow_packets_s = total_packets / duration if duration > 0 else 0
        
        return {
            'timestamp': self.generate_timestamp(),
            'duration': round(duration, 6),
            'protocol_type': protocol,
            'service': service,
            'flag': flag,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'fwd_packets': fwd_packets,
            'bwd_packets': bwd_packets,
            'total_packets': total_packets,
            'fwd_bytes': fwd_bytes,
            'bwd_bytes': bwd_bytes,
            'total_bytes': total_bytes,
            'flow_bytes_s': round(flow_bytes_s, 2),
            'flow_packets_s': round(flow_packets_s, 2),
            'label': 'Benign',
            'attack_type': 'Benign'
        }
    
    def generate_dos_attack(self):
        """Generate DoS attack patterns"""
        features = self.generate_benign_traffic()
        
        # DoS characteristics: high volume, short duration
        features['duration'] = round(random.uniform(0.001, 1.0), 6)
        features['total_packets'] = random.randint(1000, 10000)
        features['total_bytes'] = features['total_packets'] * random.randint(100, 500)
        features['flow_bytes_s'] = random.uniform(100000, 1000000)
        features['flow_packets_s'] = random.uniform(1000, 10000)
        features['flag'] = 'S0'  # SYN flood
        features['label'] = 'Attack'
        features['attack_type'] = 'DoS'
        
        return features
    
    def generate_portscan_attack(self):
        """Generate PortScan attack patterns"""
        features = self.generate_benign_traffic()
        
        # PortScan: many connections to different ports
        features['duration'] = round(random.uniform(1, 60), 6)
        features['dst_port'] = random.randint(1, 1024)  # Well-known ports
        features['total_packets'] = random.randint(3, 10)
        features['total_bytes'] = random.randint(100, 500)
        features['flag'] = 'REJ'  # Connection rejected
        features['label'] = 'Attack'
        features['attack_type'] = 'PortScan'
        
        return features
    
    def generate_bruteforce_attack(self):
        """Generate BruteForce attack patterns"""
        features = self.generate_benign_traffic()
        
        # BruteForce: multiple failed logins
        features['service'] = 'SSH'
        features['dst_port'] = 22
        features['duration'] = round(random.uniform(1, 10), 6)
        features['fwd_packets'] = random.randint(10, 50)
        features['total_packets'] = features['fwd_packets'] + random.randint(1, 5)
        features['flag'] = 'S0'
        features['label'] = 'Attack'
        features['attack_type'] = 'BruteForce'
        
        return features
    
    def generate_dataset(self, n_records=10000, attack_ratio=0.2):
        """
        Generate complete dataset with mixed traffic
        
        Parameters:
        - n_records: Total number of records to generate
        - attack_ratio: Proportion of attack traffic (0.2 = 20%)
        """
        print(f"Generating {n_records} synthetic network records...")
        print(f"Attack ratio: {attack_ratio*100}%")
        
        n_attacks = int(n_records * attack_ratio)
        n_benign = n_records - n_attacks
        
        data = []
        
        # Generate benign traffic
        print("Generating benign traffic...")
        for _ in range(n_benign):
            data.append(self.generate_benign_traffic())
        
        # Generate attack traffic
        print("Generating attack traffic...")
        attack_generators = [
            self.generate_dos_attack,
            self.generate_portscan_attack,
            self.generate_bruteforce_attack
        ]
        
        attack_weights = [0.5, 0.3, 0.2]  # DoS most common
        
        for _ in range(n_attacks):
            generator = np.random.choice(attack_generators, p=attack_weights)
            data.append(generator())
        
        # Shuffle data
        random.shuffle(data)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add flow ID
        df['flow_id'] = range(1, len(df) + 1)
        
        # Reorder columns
        column_order = [
            'flow_id', 'timestamp', 'duration', 'protocol_type', 'service', 'flag',
            'src_ip', 'dst_ip', 'src_port', 'dst_port',
            'fwd_packets', 'bwd_packets', 'total_packets',
            'fwd_bytes', 'bwd_bytes', 'total_bytes',
            'flow_bytes_s', 'flow_packets_s',
            'label', 'attack_type'
        ]
        
        df = df[column_order]
        
        print(f"\nDataset generated successfully!")
        print(f"Total records: {len(df)}")
        print(f"Benign: {len(df[df['label'] == 'Benign'])}")
        print(f"Attacks: {len(df[df['label'] == 'Attack'])}")
        print(f"\nAttack distribution:")
        print(df['attack_type'].value_counts())
        
        return df
    
    def save_dataset(self, df, output_dir='./synthetic_data'):
        """Save dataset to CSV and provide statistics"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_network_data_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"\nDataset saved to: {filepath}")
        
        # Generate statistics file
        stats_file = os.path.join(output_dir, f"data_statistics_{timestamp}.txt")
        with open(stats_file, 'w') as f:
            f.write("THE ARZENS Synthetic Network Data Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Features: {len(df.columns)}\n\n")
            f.write("Label Distribution:\n")
            f.write(df['label'].value_counts().to_string() + "\n\n")
            f.write("Attack Type Distribution:\n")
            f.write(df['attack_type'].value_counts().to_string() + "\n\n")
            f.write("Protocol Distribution:\n")
            f.write(df['protocol_type'].value_counts().to_string() + "\n\n")
            f.write("Numerical Features Summary:\n")
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            f.write(df[numerical_cols].describe().to_string() + "\n")
        
        print(f"Statistics saved to: {stats_file}")
        return filepath


def main():
    parser = argparse.ArgumentParser(
        description='THE ARZENS Synthetic Network Data Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_synthetic_network_data.py --records 10000 --attacks 0.2
  python generate_synthetic_network_data.py --records 50000 --attacks 0.3 --output ./my_data
  python generate_synthetic_network_data.py --records 1000 --seed 123
        """
    )
    
    parser.add_argument(
        '--records', '-n',
        type=int,
        default=10000,
        help='Number of records to generate (default: 10000)'
    )
    
    parser.add_argument(
        '--attacks', '-a',
        type=float,
        default=0.2,
        help='Attack traffic ratio 0.0-1.0 (default: 0.2 = 20%)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./synthetic_data',
        help='Output directory (default: ./synthetic_data)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.attacks < 0 or args.attacks > 1:
        print("Error: Attack ratio must be between 0.0 and 1.0")
        return
    
    if args.records < 100:
        print("Error: Minimum 100 records required")
        return
    
    # Generate dataset
    generator = NetworkDataGenerator(random_seed=args.seed)
    df = generator.generate_dataset(n_records=args.records, attack_ratio=args.attacks)
    
    # Save dataset
    filepath = generator.save_dataset(df, output_dir=args.output)
    
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE!")
    print("="*50)
    print(f"\nFile location: {filepath}")
    print("\nThis dataset is ready for:")
    print("  - Assignment 4: Machine Learning for Threat Detection")
    print("  - Feature engineering and preprocessing")
    print("  - Model training and evaluation")
    print("\nFormat: CSV with 20 features + label + attack_type")
    print("Compatible with: scikit-learn, pandas, TensorFlow, PyTorch")


if __name__ == "__main__":
    main()