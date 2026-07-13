import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RemoveButton } from './remove-button';

describe('RemoveButton', () => {
  let component: RemoveButton;
  let fixture: ComponentFixture<RemoveButton>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RemoveButton],
    }).compileComponents();

    fixture = TestBed.createComponent(RemoveButton);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
